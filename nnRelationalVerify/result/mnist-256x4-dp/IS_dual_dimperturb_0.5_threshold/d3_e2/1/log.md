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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0113332, 0.0041803, -0.0113332, 0.0041803, -0.0149349, 0.0149349)
1: (-0.0027807, 0.0077282, -0.0027807, 0.0077282, -0.0105089, 0.0105089)
2: (0.0047662, 0.0433445, 0.0047662, 0.0433445, -0.0379917, 0.0379918)
3: (-0.0070429, 0.0120681, -0.0070429, 0.0120681, -0.0185035, 0.0185035)
4: (-0.0108760, 0.0218021, -0.0108760, 0.0218021, -0.0326781, 0.0326781)
5: (0.0006645, 0.0120628, 0.0006645, 0.0120628, -0.0113983, 0.0113983)
6: (0.0001577, 0.0125902, 0.0001577, 0.0125902, -0.0124324, 0.0124324)
7: (-0.0351038, -0.0003710, -0.0351038, -0.0003710, -0.0302181, 0.0302181)
8: (0.9498612, 1.0227283, 0.9498612, 1.0227283, -0.0728672, 0.0728672)
9: (-0.0098954, 0.0098825, -0.0098954, 0.0098825, -0.0197779, 0.0197779)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.81 = 3.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0383648, upper bound: 0.0383648

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0376038, upper bound: 0.0374199
time: 1.19 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0375457, upper bound: 0.0375457
time: 0.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.29 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 8, lower bound: -0.0376038, upper bound: 0.0374199
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 8, lower bound: -0.0375457, upper bound: 0.0375457

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0113332, 0.0041803, -0.0028019, 0.0041757, -0.0149227, 0.0064030
1: -0.0027807, 0.0077282, -0.0011204, 0.0077211, -0.0105018, 0.0088486
2: 0.0047662, 0.0433445, 0.0047769, 0.0227735, -0.0166429, 0.0379761
3: -0.0070429, 0.0120681, -0.0070349, 0.0043011, -0.0109681, 0.0184917
4: -0.0108760, 0.0218021, -0.0108686, 0.0029851, -0.0138612, 0.0326707
5: 0.0006645, 0.0120628, 0.0008675, 0.0106441, -0.0099795, 0.0110448
6: 0.0001577, 0.0125902, 0.0001685, 0.0125872, -0.0124294, 0.0124216
7: -0.0351038, -0.0003710, -0.0234731, -0.0003883, -0.0301925, 0.0187148
8: 0.9498612, 1.0227283, 0.9614691, 1.0226784, -0.0728172, 0.0612593
9: -0.0098954, 0.0098825, -0.0091547, 0.0082967, -0.0181921, 0.0172075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0374199, upper bound: 0.0374199
time: 1.02 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0374199, upper bound: 0.0374199
time: 0.92 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0101340, 0.0041795, -0.0040130, 0.0043135, -0.0138696, 0.0076292
1: -0.0025983, 0.0077268, -0.0014131, 0.0079323, -0.0105305, 0.0091400
2: 0.0047682, 0.0404416, 0.0044606, 0.0256516, -0.0194370, 0.0352497
3: -0.0070414, 0.0109631, -0.0072727, 0.0053643, -0.0117392, 0.0175427
4: -0.0108746, 0.0191480, -0.0110880, 0.0056374, -0.0165120, 0.0302361
5: 0.0006998, 0.0119039, 0.0006013, 0.0108835, -0.0101837, 0.0113026
6: 0.0001598, 0.0125896, -0.0001518, 0.0126767, -0.0125170, 0.0127414
7: -0.0334442, -0.0003742, -0.0250460, 0.0001270, -0.0290492, 0.0203740
8: 0.9516704, 1.0227188, 0.9596804, 1.0241549, -0.0724845, 0.0630383
9: -0.0097264, 0.0096959, -0.0096767, 0.0085325, -0.0182589, 0.0180910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338052, upper bound: 0.0356413
time: 0.92 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371380, upper bound: 0.0371380
time: 1.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.48 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 8, lower bound: -0.0374199, upper bound: 0.0374199
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 8, lower bound: -0.0374199, upper bound: 0.0374199
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 8, lower bound: -0.0338052, upper bound: 0.0356413
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 8, lower bound: -0.0371380, upper bound: 0.0371380

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028019, 0.0041757, -0.0028019, 0.0041757, -0.0063908, 0.0063908
1: -0.0011204, 0.0077211, -0.0011204, 0.0077211, -0.0088414, 0.0088414
2: 0.0047769, 0.0227735, 0.0047769, 0.0227735, -0.0166272, 0.0166272
3: -0.0070349, 0.0043011, -0.0070349, 0.0043011, -0.0109563, 0.0109563
4: -0.0108686, 0.0029851, -0.0108686, 0.0029851, -0.0138538, 0.0138538
5: 0.0008675, 0.0106441, 0.0008675, 0.0106441, -0.0097766, 0.0097766
6: 0.0001685, 0.0125872, 0.0001685, 0.0125872, -0.0124186, 0.0124186
7: -0.0234731, -0.0003883, -0.0234731, -0.0003883, -0.0186893, 0.0186893
8: 0.9614691, 1.0226784, 0.9614691, 1.0226784, -0.0612093, 0.0612093
9: -0.0091547, 0.0082967, -0.0091547, 0.0082967, -0.0165501, 0.0165501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338981, upper bound: 0.0355104
time: 0.85 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371341, upper bound: 0.0369955
time: 0.90 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040130, 0.0043135, -0.0028019, 0.0041757, -0.0076142, 0.0065367
1: -0.0014131, 0.0079323, -0.0011204, 0.0077211, -0.0091342, 0.0090526
2: 0.0044606, 0.0256516, 0.0047769, 0.0227735, -0.0169644, 0.0193614
3: -0.0072727, 0.0053643, -0.0070349, 0.0043011, -0.0112099, 0.0116464
4: -0.0110880, 0.0056374, -0.0108686, 0.0029851, -0.0140732, 0.0165060
5: 0.0006013, 0.0108835, 0.0008675, 0.0106441, -0.0100428, 0.0100160
6: -0.0001518, 0.0126767, 0.0001685, 0.0125872, -0.0127389, 0.0125082
7: -0.0250460, 0.0001270, -0.0234731, -0.0003883, -0.0202146, 0.0192388
8: 0.9596804, 1.0241549, 0.9614691, 1.0226784, -0.0629979, 0.0626858
9: -0.0096767, 0.0085325, -0.0091547, 0.0082967, -0.0175530, 0.0165349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357057, upper bound: 0.0338418
time: 0.98 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0371341, upper bound: 0.0369955
time: 1.29 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003276, 0.0043549, -0.0013930, 0.0043126, -0.0041336, 0.0051677
1: -0.0002136, 0.0079956, -0.0006908, 0.0079308, -0.0081444, 0.0086864
2: 0.0043657, 0.0166599, 0.0044627, 0.0193572, -0.0149915, 0.0121972
3: -0.0073441, 0.0019006, -0.0072711, 0.0029972, -0.0103413, 0.0091718
4: -0.0111538, -0.0015266, -0.0110865, -0.0001359, -0.0110179, 0.0095600
5: 0.0006075, 0.0098353, 0.0006593, 0.0102733, -0.0096659, 0.0091760
6: -0.0002479, 0.0127036, -0.0001496, 0.0126761, -0.0129240, 0.0128532
7: -0.0197509, 0.0002817, -0.0214976, 0.0001235, -0.0160938, 0.0176654
8: 0.9672022, 1.0245980, 0.9641494, 1.0241450, -0.0569428, 0.0604486
9: -0.0095763, 0.0072924, -0.0095048, 0.0078717, -0.0161561, 0.0158534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0316459, upper bound: 0.0338827
time: 1.16 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0316241, upper bound: 0.0337366
time: 0.91 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0068190, 0.0041778, -0.0040130, 0.0043135, -0.0105403, 0.0076249
1: -0.0020030, 0.0077243, -0.0014131, 0.0079323, -0.0099352, 0.0091374
2: 0.0047720, 0.0324002, 0.0044606, 0.0256516, -0.0194326, 0.0264971
3: -0.0070385, 0.0079031, -0.0072727, 0.0053643, -0.0117359, 0.0141101
4: -0.0108720, 0.0118122, -0.0110880, 0.0056374, -0.0165093, 0.0229002
5: 0.0007812, 0.0113855, 0.0006013, 0.0108835, -0.0101023, 0.0107843
6: 0.0001636, 0.0125885, -0.0001518, 0.0126767, -0.0125131, 0.0127403
7: -0.0288338, -0.0003804, -0.0250460, 0.0001270, -0.0241545, 0.0203667
8: 0.9560204, 1.0227011, 0.9596804, 1.0241549, -0.0681345, 0.0630207
9: -0.0094321, 0.0090879, -0.0096767, 0.0085325, -0.0179646, 0.0178015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369955, upper bound: 0.0371380
time: 1.19 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0369955, upper bound: 0.0370055
time: 0.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.70 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 8, lower bound: -0.0338981, upper bound: 0.0355104
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 8, lower bound: -0.0371341, upper bound: 0.0369955
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 8, lower bound: -0.0357057, upper bound: 0.0338418
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 8, lower bound: -0.0371341, upper bound: 0.0369955
IS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.70
Output dim: 8, lower bound: -0.0316459, upper bound: 0.0338827
IS_B2_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.70
Output dim: 8, lower bound: -0.0316241, upper bound: 0.0337366
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 8, lower bound: -0.0369955, upper bound: 0.0371380
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 8, lower bound: -0.0369955, upper bound: 0.0370055

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002942, 0.0043512, -0.0003609, 0.0041747, -0.0039565, 0.0041919
1: -0.0000570, 0.0079900, -0.0003691, 0.0077196, -0.0077766, 0.0083591
2: 0.0043741, 0.0164254, 0.0047791, 0.0168928, -0.0125187, 0.0116463
3: -0.0073377, 0.0017243, -0.0070333, 0.0020758, -0.0094135, 0.0087575
4: -0.0111480, -0.0015288, -0.0108671, -0.0016380, -0.0095099, 0.0093383
5: 0.0006138, 0.0096593, 0.0009177, 0.0100102, -0.0093964, 0.0087416
6: -0.0002393, 0.0127012, 0.0001708, 0.0125865, -0.0128258, 0.0125305
7: -0.0193687, 0.0002679, -0.0201304, -0.0003919, -0.0152180, 0.0164844
8: 0.9682971, 1.0245587, 0.9661148, 1.0226682, -0.0543711, 0.0584439
9: -0.0095647, 0.0069706, -0.0090091, 0.0076120, -0.0157848, 0.0148713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0322130, upper bound: 0.0342784
time: 1.18 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0322021, upper bound: 0.0341994
time: 0.91 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0003522, 0.0041741, -0.0028019, 0.0041757, -0.0040176, 0.0063863
1: -0.0003286, 0.0077186, -0.0011204, 0.0077211, -0.0080497, 0.0088389
2: 0.0047806, 0.0168321, 0.0047769, 0.0227735, -0.0166228, 0.0120552
3: -0.0070321, 0.0020301, -0.0070349, 0.0043011, -0.0109531, 0.0090650
4: -0.0108660, -0.0016385, -0.0108686, 0.0029851, -0.0138512, 0.0092302
5: 0.0009189, 0.0099646, 0.0008675, 0.0106441, -0.0097252, 0.0090971
6: 0.0001723, 0.0125861, 0.0001685, 0.0125872, -0.0124148, 0.0124175
7: -0.0200315, -0.0003944, -0.0234731, -0.0003883, -0.0157748, 0.0186821
8: 0.9663982, 1.0226610, 0.9614691, 1.0226784, -0.0562801, 0.0611919
9: -0.0090069, 0.0075287, -0.0091547, 0.0082967, -0.0156021, 0.0162754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357249, upper bound: 0.0358176
time: 0.96 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357231, upper bound: 0.0357231
time: 0.98 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0013930, 0.0043126, -0.0002942, 0.0043512, -0.0051555, 0.0041027
1: -0.0006908, 0.0079308, -0.0000570, 0.0079900, -0.0086808, 0.0079878
2: 0.0044627, 0.0193572, 0.0043741, 0.0164254, -0.0119626, 0.0149831
3: -0.0072711, 0.0029972, -0.0073377, 0.0017243, -0.0089954, 0.0103349
4: -0.0110865, -0.0001359, -0.0111480, -0.0015288, -0.0095577, 0.0110121
5: 0.0006593, 0.0102733, 0.0006138, 0.0096593, -0.0090000, 0.0096596
6: -0.0001496, 0.0126761, -0.0002393, 0.0127012, -0.0128508, 0.0129154
7: -0.0214976, 0.0001235, -0.0193687, 0.0002679, -0.0174740, 0.0157678
8: 0.9641494, 1.0241450, 0.9682971, 1.0245587, -0.0604092, 0.0558479
9: -0.0095048, 0.0078717, -0.0095647, 0.0069706, -0.0156071, 0.0157413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0341746, upper bound: 0.0319752
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340404, upper bound: 0.0319198
time: 0.88 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040130, 0.0043135, -0.0003522, 0.0041741, -0.0076097, 0.0041649
1: -0.0014131, 0.0079323, -0.0003286, 0.0077186, -0.0091317, 0.0082609
2: 0.0044606, 0.0256516, 0.0047806, 0.0168321, -0.0123715, 0.0193570
3: -0.0072727, 0.0053643, -0.0070321, 0.0020301, -0.0093029, 0.0116431
4: -0.0110880, 0.0056374, -0.0108660, -0.0016385, -0.0094495, 0.0165034
5: 0.0006013, 0.0108835, 0.0009189, 0.0099646, -0.0093633, 0.0099646
6: -0.0001518, 0.0126767, 0.0001723, 0.0125861, -0.0127379, 0.0125044
7: -0.0250460, 0.0001270, -0.0200315, -0.0003944, -0.0202074, 0.0162880
8: 0.9596804, 1.0241549, 0.9663982, 1.0226610, -0.0629805, 0.0577567
9: -0.0096767, 0.0085325, -0.0090069, 0.0075287, -0.0172054, 0.0155869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0358706, upper bound: 0.0355427
time: 0.97 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356805, upper bound: 0.0355212
time: 0.83 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0003522, 0.0041741, -0.0040130, 0.0043135, -0.0041649, 0.0076097
1: -0.0003286, 0.0077186, -0.0014131, 0.0079323, -0.0082609, 0.0091317
2: 0.0047806, 0.0168321, 0.0044606, 0.0256516, -0.0193570, 0.0123715
3: -0.0070321, 0.0020301, -0.0072727, 0.0053643, -0.0116431, 0.0093029
4: -0.0108660, -0.0016385, -0.0110880, 0.0056374, -0.0165034, 0.0094495
5: 0.0009189, 0.0099646, 0.0006013, 0.0108835, -0.0099646, 0.0093633
6: 0.0001723, 0.0125861, -0.0001518, 0.0126767, -0.0125044, 0.0127379
7: -0.0200315, -0.0003944, -0.0250460, 0.0001270, -0.0162880, 0.0202074
8: 0.9663982, 1.0226610, 0.9596804, 1.0241549, -0.0577567, 0.0629805
9: -0.0090069, 0.0075287, -0.0096767, 0.0085325, -0.0155869, 0.0172054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355426, upper bound: 0.0357994
time: 0.98 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355200, upper bound: 0.0355647
time: 0.95 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0008122, 0.0043118, -0.0040130, 0.0043135, -0.0045366, 0.0077105
1: -0.0004813, 0.0079297, -0.0014131, 0.0079323, -0.0084135, 0.0093428
2: 0.0044644, 0.0179383, 0.0044606, 0.0256516, -0.0194339, 0.0134778
3: -0.0072699, 0.0024468, -0.0072727, 0.0053643, -0.0117369, 0.0097196
4: -0.0110854, -0.0013734, -0.0110880, 0.0056374, -0.0167227, 0.0097146
5: 0.0006723, 0.0100906, 0.0006013, 0.0108835, -0.0102112, 0.0094893
6: -0.0001479, 0.0126757, -0.0001518, 0.0126767, -0.0128246, 0.0128274
7: -0.0206528, 0.0001208, -0.0250460, 0.0001270, -0.0163557, 0.0203690
8: 0.9654623, 1.0241371, 0.9596804, 1.0241549, -0.0586926, 0.0644567
9: -0.0094678, 0.0076562, -0.0096767, 0.0085325, -0.0161010, 0.0173329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355104, upper bound: 0.0338025
time: 1.01 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355104, upper bound: 0.0369985
time: 1.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.56 seconds
IS_B1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0322130, upper bound: 0.0342784
IS_B1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0322021, upper bound: 0.0341994
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0357249, upper bound: 0.0358176
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0357231, upper bound: 0.0357231
IS_B1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0341746, upper bound: 0.0319752
IS_B1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0340404, upper bound: 0.0319198
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0358706, upper bound: 0.0355427
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0356805, upper bound: 0.0355212
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0355426, upper bound: 0.0357994
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0355200, upper bound: 0.0355647
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0355104, upper bound: 0.0338025
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.56
Output dim: 8, lower bound: -0.0355104, upper bound: 0.0369985

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003522, 0.0041741, -0.0003177, 0.0041692, -0.0040077, 0.0039813
1: -0.0003286, 0.0077186, -0.0001669, 0.0077111, -0.0080397, 0.0078855
2: 0.0047806, 0.0168321, 0.0047917, 0.0165899, -0.0118093, 0.0120404
3: -0.0070321, 0.0020301, -0.0070237, 0.0018480, -0.0088801, 0.0090539
4: -0.0108660, -0.0016385, -0.0108583, -0.0016415, -0.0092246, 0.0092199
5: 0.0009189, 0.0099646, 0.0009272, 0.0097828, -0.0088639, 0.0090374
6: 0.0001723, 0.0125861, 0.0001836, 0.0125829, -0.0124106, 0.0124025
7: -0.0200315, -0.0003944, -0.0196369, -0.0004126, -0.0157560, 0.0154789
8: 0.9663982, 1.0226610, 0.9675288, 1.0226090, -0.0562108, 0.0551322
9: -0.0090069, 0.0075287, -0.0089917, 0.0071964, -0.0149820, 0.0152380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356761, upper bound: 0.0356761
time: 1.05 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356761, upper bound: 0.0356761
time: 1.22 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003372, 0.0041721, -0.0002938, 0.0043997, -0.0042581, 0.0039859
1: -0.0002583, 0.0077156, -0.0000553, 0.0080644, -0.0083227, 0.0077709
2: 0.0047851, 0.0167269, 0.0042627, 0.0164228, -0.0116378, 0.0124642
3: -0.0070287, 0.0019510, -0.0074216, 0.0017224, -0.0087511, 0.0093726
4: -0.0108630, -0.0016397, -0.0112253, -0.0014988, -0.0093642, 0.0095856
5: 0.0009222, 0.0098856, 0.0005301, 0.0096574, -0.0087352, 0.0093555
6: 0.0001768, 0.0125848, -0.0003522, 0.0127328, -0.0125560, 0.0129370
7: -0.0198600, -0.0004017, -0.0193646, 0.0004495, -0.0168430, 0.0155614
8: 0.9668895, 1.0226402, 0.9683089, 1.0250790, -0.0581895, 0.0543314
9: -0.0090009, 0.0073843, -0.0097176, 0.0069672, -0.0151558, 0.0161541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356761, upper bound: 0.0357231
time: 0.87 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356761, upper bound: 0.0357231
time: 0.98 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003190, 0.0043069, -0.0003522, 0.0041741, -0.0039901, 0.0041554
1: -0.0001733, 0.0079222, -0.0003286, 0.0077186, -0.0078918, 0.0082508
2: 0.0044757, 0.0165995, 0.0047806, 0.0168321, -0.0123564, 0.0118188
3: -0.0072614, 0.0018552, -0.0070321, 0.0020301, -0.0092915, 0.0088873
4: -0.0110775, -0.0015562, -0.0108660, -0.0016385, -0.0094391, 0.0093098
5: 0.0006900, 0.0097900, 0.0009189, 0.0099646, -0.0092746, 0.0088711
6: -0.0001365, 0.0126725, 0.0001723, 0.0125861, -0.0127225, 0.0125001
7: -0.0196524, 0.0001024, -0.0200315, -0.0003944, -0.0154832, 0.0162682
8: 0.9674844, 1.0240846, 0.9663982, 1.0226610, -0.0551766, 0.0576863
9: -0.0094253, 0.0072095, -0.0090069, 0.0075287, -0.0156693, 0.0148907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356602, upper bound: 0.0354666
time: 1.15 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356602, upper bound: 0.0355212
time: 1.28 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002910, 0.0045291, -0.0003372, 0.0041721, -0.0039914, 0.0043836
1: -0.0000421, 0.0082627, -0.0002583, 0.0077156, -0.0077577, 0.0085210
2: 0.0039658, 0.0164031, 0.0047851, 0.0167269, -0.0127611, 0.0116181
3: -0.0076448, 0.0017075, -0.0070287, 0.0019510, -0.0095958, 0.0087363
4: -0.0114312, -0.0014187, -0.0108630, -0.0016397, -0.0097915, 0.0094443
5: 0.0003073, 0.0096426, 0.0009222, 0.0098856, -0.0095783, 0.0087204
6: -0.0006528, 0.0128169, 0.0001768, 0.0125848, -0.0132377, 0.0126401
7: -0.0193325, 0.0009333, -0.0198600, -0.0004017, -0.0155528, 0.0172751
8: 0.9684011, 1.0264651, 0.9668895, 1.0226402, -0.0542392, 0.0595756
9: -0.0101250, 0.0069401, -0.0090009, 0.0073843, -0.0165179, 0.0150615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356805, upper bound: 0.0354666
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356805, upper bound: 0.0355212
time: 0.99 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003522, 0.0041741, -0.0003190, 0.0043069, -0.0041554, 0.0039901
1: -0.0003286, 0.0077186, -0.0001733, 0.0079222, -0.0082508, 0.0078918
2: 0.0047806, 0.0168321, 0.0044757, 0.0165995, -0.0118188, 0.0123564
3: -0.0070321, 0.0020301, -0.0072614, 0.0018552, -0.0088873, 0.0092915
4: -0.0108660, -0.0016385, -0.0110775, -0.0015562, -0.0093098, 0.0094391
5: 0.0009189, 0.0099646, 0.0006900, 0.0097900, -0.0088711, 0.0092746
6: 0.0001723, 0.0125861, -0.0001365, 0.0126725, -0.0125001, 0.0127225
7: -0.0200315, -0.0003944, -0.0196524, 0.0001024, -0.0162682, 0.0154832
8: 0.9663982, 1.0226610, 0.9674844, 1.0240846, -0.0576863, 0.0551766
9: -0.0090069, 0.0075287, -0.0094253, 0.0072095, -0.0148907, 0.0156693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354666, upper bound: 0.0356602
time: 1.05 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354666, upper bound: 0.0356602
time: 1.02 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003372, 0.0041721, -0.0002910, 0.0045291, -0.0043836, 0.0039914
1: -0.0002583, 0.0077156, -0.0000421, 0.0082627, -0.0085210, 0.0077577
2: 0.0047851, 0.0167269, 0.0039658, 0.0164031, -0.0116181, 0.0127611
3: -0.0070287, 0.0019510, -0.0076448, 0.0017075, -0.0087363, 0.0095958
4: -0.0108630, -0.0016397, -0.0114312, -0.0014187, -0.0094443, 0.0097915
5: 0.0009222, 0.0098856, 0.0003073, 0.0096426, -0.0087204, 0.0095783
6: 0.0001768, 0.0125848, -0.0006528, 0.0128169, -0.0126401, 0.0132377
7: -0.0198600, -0.0004017, -0.0193325, 0.0009333, -0.0172751, 0.0155528
8: 0.9668895, 1.0226402, 0.9684011, 1.0264651, -0.0595756, 0.0542392
9: -0.0090009, 0.0073843, -0.0101250, 0.0069401, -0.0150615, 0.0165180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354666, upper bound: 0.0356805
time: 1.16 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354666, upper bound: 0.0356805
time: 1.26 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008122, 0.0043118, -0.0002929, 0.0044939, -0.0047143, 0.0040694
1: -0.0004813, 0.0079297, -0.0000508, 0.0082087, -0.0086900, 0.0079805
2: 0.0044644, 0.0179383, 0.0040466, 0.0164161, -0.0119517, 0.0138917
3: -0.0072699, 0.0024468, -0.0075840, 0.0017173, -0.0089872, 0.0100309
4: -0.0110854, -0.0013734, -0.0113751, -0.0014405, -0.0096449, 0.0100018
5: 0.0006723, 0.0100906, 0.0003680, 0.0096523, -0.0089801, 0.0097226
6: -0.0001479, 0.0126757, -0.0005710, 0.0127940, -0.0129419, 0.0132466
7: -0.0206528, 0.0001208, -0.0193536, 0.0008016, -0.0171307, 0.0154924
8: 0.9654623, 1.0241371, 0.9683406, 1.0260875, -0.0606253, 0.0557966
9: -0.0094678, 0.0076562, -0.0100140, 0.0069579, -0.0153393, 0.0160853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338827, upper bound: 0.0316459
time: 0.80 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0337366, upper bound: 0.0316241
time: 0.94 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008122, 0.0043118, -0.0008122, 0.0043118, -0.0045322, 0.0045322
1: -0.0004813, 0.0079297, -0.0004813, 0.0079297, -0.0084110, 0.0084110
2: 0.0044644, 0.0179383, 0.0044644, 0.0179383, -0.0134739, 0.0134739
3: -0.0072699, 0.0024468, -0.0072699, 0.0024468, -0.0097167, 0.0097167
4: -0.0110854, -0.0013734, -0.0110854, -0.0013734, -0.0097120, 0.0097120
5: 0.0006723, 0.0100906, 0.0006723, 0.0100906, -0.0094183, 0.0094183
6: -0.0001479, 0.0126757, -0.0001479, 0.0126757, -0.0128235, 0.0128235
7: -0.0206528, 0.0001208, -0.0206528, 0.0001208, -0.0163480, 0.0163480
8: 0.9654623, 1.0241371, 0.9654623, 1.0241371, -0.0586749, 0.0586749
9: -0.0094678, 0.0076562, -0.0094678, 0.0076562, -0.0157214, 0.0157214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338402, upper bound: 0.0359731
time: 1.00 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0336815, upper bound: 0.0351830
time: 1.29 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.89 seconds
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0356761, upper bound: 0.0356761
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0356761, upper bound: 0.0356761
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0356761, upper bound: 0.0357231
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0356761, upper bound: 0.0357231
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0356602, upper bound: 0.0354666
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0356602, upper bound: 0.0355212
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0356805, upper bound: 0.0354666
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0356805, upper bound: 0.0355212
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0354666, upper bound: 0.0356602
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0354666, upper bound: 0.0356602
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0354666, upper bound: 0.0356805
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0354666, upper bound: 0.0356805
IS_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0338827, upper bound: 0.0316459
IS_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0337366, upper bound: 0.0316241
IS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0338402, upper bound: 0.0359731
IS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.89
Output dim: 8, lower bound: -0.0336815, upper bound: 0.0351830

## BFS IS instance: IS_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002958, 0.0041676, -0.0003177, 0.0041692, -0.0039451, 0.0039741
1: -0.0000646, 0.0077086, -0.0001669, 0.0077111, -0.0077758, 0.0078755
2: 0.0047955, 0.0164368, 0.0047917, 0.0165899, -0.0117944, 0.0116450
3: -0.0070209, 0.0017329, -0.0070237, 0.0018480, -0.0088689, 0.0087566
4: -0.0108557, -0.0016425, -0.0108583, -0.0016415, -0.0092142, 0.0092158
5: 0.0009301, 0.0096679, 0.0009272, 0.0097828, -0.0088527, 0.0087406
6: 0.0001875, 0.0125819, 0.0001836, 0.0125829, -0.0123955, 0.0123983
7: -0.0193873, -0.0004188, -0.0196369, -0.0004126, -0.0151331, 0.0154587
8: 0.9682440, 1.0225914, 0.9675288, 1.0226090, -0.0543650, 0.0550625
9: -0.0089865, 0.0069863, -0.0089917, 0.0071964, -0.0149649, 0.0147493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355233, upper bound: 0.0355390
time: 0.92 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355233, upper bound: 0.0355694
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002727, 0.0043981, -0.0003177, 0.0041692, -0.0039539, 0.0042354
1: 0.0000435, 0.0080620, -0.0001669, 0.0077111, -0.0076676, 0.0082289
2: 0.0042663, 0.0162748, 0.0047917, 0.0165899, -0.0123236, 0.0114831
3: -0.0074188, 0.0016111, -0.0070237, 0.0018480, -0.0092668, 0.0086348
4: -0.0112227, -0.0014998, -0.0108583, -0.0016415, -0.0095813, 0.0093586
5: 0.0005329, 0.0095463, 0.0009272, 0.0097828, -0.0092499, 0.0086191
6: -0.0003485, 0.0127318, 0.0001836, 0.0125829, -0.0129314, 0.0125482
7: -0.0191234, 0.0004435, -0.0196369, -0.0004126, -0.0150866, 0.0167253
8: 0.9689999, 1.0250618, 0.9675288, 1.0226090, -0.0536091, 0.0575330
9: -0.0097126, 0.0067641, -0.0089917, 0.0071964, -0.0160315, 0.0146280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355233, upper bound: 0.0355390
time: 1.05 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355233, upper bound: 0.0355694
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002958, 0.0041676, -0.0002938, 0.0043997, -0.0042070, 0.0039903
1: -0.0000646, 0.0077086, -0.0000553, 0.0080644, -0.0081290, 0.0077639
2: 0.0047955, 0.0164368, 0.0042627, 0.0164228, -0.0116273, 0.0121741
3: -0.0070209, 0.0017329, -0.0074216, 0.0017224, -0.0087432, 0.0091544
4: -0.0108557, -0.0016425, -0.0112253, -0.0014988, -0.0093569, 0.0095828
5: 0.0009301, 0.0096679, 0.0005301, 0.0096574, -0.0087273, 0.0091377
6: 0.0001875, 0.0125819, -0.0003522, 0.0127328, -0.0125453, 0.0129340
7: -0.0193873, -0.0004188, -0.0193646, 0.0004495, -0.0163664, 0.0155221
8: 0.9682440, 1.0225914, 0.9683089, 1.0250790, -0.0568351, 0.0542825
9: -0.0089865, 0.0069863, -0.0097176, 0.0069672, -0.0149229, 0.0157879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354733, upper bound: 0.0355719
time: 0.82 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354982, upper bound: 0.0355719
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002727, 0.0043981, -0.0002938, 0.0043997, -0.0041210, 0.0041513
1: 0.0000435, 0.0080620, -0.0000553, 0.0080644, -0.0080209, 0.0081173
2: 0.0042663, 0.0162748, 0.0042627, 0.0164228, -0.0121565, 0.0120122
3: -0.0074188, 0.0016111, -0.0074216, 0.0017224, -0.0091412, 0.0090326
4: -0.0112227, -0.0014998, -0.0112253, -0.0014988, -0.0097240, 0.0097255
5: 0.0005329, 0.0095463, 0.0005301, 0.0096574, -0.0091245, 0.0090162
6: -0.0003485, 0.0127318, -0.0003522, 0.0127328, -0.0130813, 0.0130839
7: -0.0191234, 0.0004435, -0.0193646, 0.0004495, -0.0153412, 0.0156798
8: 0.9689999, 1.0250618, 0.9683089, 1.0250790, -0.0560791, 0.0567530
9: -0.0097126, 0.0067641, -0.0097176, 0.0069672, -0.0152555, 0.0150469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354982, upper bound: 0.0354733
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354982, upper bound: 0.0354982
time: 1.02 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003190, 0.0043069, -0.0002958, 0.0041676, -0.0039830, 0.0040928
1: -0.0001733, 0.0079222, -0.0000646, 0.0077086, -0.0078818, 0.0079868
2: 0.0044757, 0.0165995, 0.0047955, 0.0164368, -0.0119611, 0.0118039
3: -0.0072614, 0.0018552, -0.0070209, 0.0017329, -0.0089942, 0.0088761
4: -0.0110775, -0.0015562, -0.0108557, -0.0016425, -0.0094350, 0.0092994
5: 0.0006900, 0.0097900, 0.0009301, 0.0096679, -0.0089778, 0.0088599
6: -0.0001365, 0.0126725, 0.0001875, 0.0125819, -0.0127183, 0.0124850
7: -0.0196524, 0.0001024, -0.0193873, -0.0004188, -0.0154629, 0.0156453
8: 0.9674844, 1.0240846, 0.9682440, 1.0225914, -0.0551069, 0.0558406
9: -0.0094253, 0.0072095, -0.0089865, 0.0069863, -0.0151807, 0.0148737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355833, upper bound: 0.0352629
time: 0.86 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356058, upper bound: 0.0352629
time: 1.08 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003190, 0.0043069, -0.0002727, 0.0043981, -0.0042443, 0.0041017
1: -0.0001733, 0.0079222, 0.0000435, 0.0080620, -0.0082352, 0.0078787
2: 0.0044757, 0.0165995, 0.0042663, 0.0162748, -0.0117991, 0.0123331
3: -0.0072614, 0.0018552, -0.0074188, 0.0016111, -0.0088725, 0.0092740
4: -0.0110775, -0.0015562, -0.0112227, -0.0014998, -0.0095778, 0.0096665
5: 0.0006900, 0.0097900, 0.0005329, 0.0095463, -0.0088563, 0.0092571
6: -0.0001365, 0.0126725, -0.0003485, 0.0127318, -0.0128682, 0.0130209
7: -0.0196524, 0.0001024, -0.0191234, 0.0004435, -0.0167295, 0.0155988
8: 0.9674844, 1.0240846, 0.9689999, 1.0250618, -0.0575774, 0.0550846
9: -0.0094253, 0.0072095, -0.0097126, 0.0067641, -0.0150593, 0.0159402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355833, upper bound: 0.0353016
time: 1.01 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356058, upper bound: 0.0353016
time: 1.11 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002910, 0.0045291, -0.0002958, 0.0041676, -0.0039856, 0.0043325
1: -0.0000421, 0.0082627, -0.0000646, 0.0077086, -0.0077507, 0.0083273
2: 0.0039658, 0.0164031, 0.0047955, 0.0164368, -0.0124710, 0.0116076
3: -0.0076448, 0.0017075, -0.0070209, 0.0017329, -0.0093777, 0.0087284
4: -0.0114312, -0.0014187, -0.0108557, -0.0016425, -0.0097887, 0.0094370
5: 0.0003073, 0.0096426, 0.0009301, 0.0096679, -0.0093606, 0.0087125
6: -0.0006528, 0.0128169, 0.0001875, 0.0125819, -0.0132347, 0.0126294
7: -0.0193325, 0.0009333, -0.0193873, -0.0004188, -0.0154425, 0.0167986
8: 0.9684011, 1.0264651, 0.9682440, 1.0225914, -0.0541903, 0.0582211
9: -0.0101250, 0.0069401, -0.0089865, 0.0069863, -0.0161518, 0.0147894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355100, upper bound: 0.0351797
time: 1.05 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355064, upper bound: 0.0351751
time: 1.41 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002910, 0.0045291, -0.0002727, 0.0043981, -0.0041569, 0.0042516
1: -0.0000421, 0.0082627, 0.0000435, 0.0080620, -0.0081041, 0.0082191
2: 0.0039658, 0.0164031, 0.0042663, 0.0162748, -0.0123091, 0.0121368
3: -0.0076448, 0.0017075, -0.0074188, 0.0016111, -0.0092559, 0.0091263
4: -0.0114312, -0.0014187, -0.0112227, -0.0014998, -0.0099314, 0.0098040
5: 0.0003073, 0.0096426, 0.0005329, 0.0095463, -0.0092390, 0.0091097
6: -0.0006528, 0.0128169, -0.0003485, 0.0127318, -0.0133846, 0.0131654
7: -0.0193325, 0.0009333, -0.0191234, 0.0004435, -0.0156712, 0.0158203
8: 0.9684011, 1.0264651, 0.9689999, 1.0250618, -0.0566608, 0.0574651
9: -0.0101250, 0.0069401, -0.0097126, 0.0067641, -0.0154503, 0.0151612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354937, upper bound: 0.0351751
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355064, upper bound: 0.0351751
time: 1.05 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002958, 0.0041676, -0.0003190, 0.0043069, -0.0040928, 0.0039830
1: -0.0000646, 0.0077086, -0.0001733, 0.0079222, -0.0079868, 0.0078818
2: 0.0047955, 0.0164368, 0.0044757, 0.0165995, -0.0118039, 0.0119611
3: -0.0070209, 0.0017329, -0.0072614, 0.0018552, -0.0088761, 0.0089942
4: -0.0108557, -0.0016425, -0.0110775, -0.0015562, -0.0092994, 0.0094350
5: 0.0009301, 0.0096679, 0.0006900, 0.0097900, -0.0088599, 0.0089778
6: 0.0001875, 0.0125819, -0.0001365, 0.0126725, -0.0124850, 0.0127183
7: -0.0193873, -0.0004188, -0.0196524, 0.0001024, -0.0156453, 0.0154629
8: 0.9682440, 1.0225914, 0.9674844, 1.0240846, -0.0558406, 0.0551069
9: -0.0089865, 0.0069863, -0.0094253, 0.0072095, -0.0148737, 0.0151807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352629, upper bound: 0.0355833
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352629, upper bound: 0.0356058
time: 1.23 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002727, 0.0043981, -0.0003190, 0.0043069, -0.0041017, 0.0042443
1: 0.0000435, 0.0080620, -0.0001733, 0.0079222, -0.0078787, 0.0082352
2: 0.0042663, 0.0162748, 0.0044757, 0.0165995, -0.0123331, 0.0117991
3: -0.0074188, 0.0016111, -0.0072614, 0.0018552, -0.0092740, 0.0088725
4: -0.0112227, -0.0014998, -0.0110775, -0.0015562, -0.0096665, 0.0095778
5: 0.0005329, 0.0095463, 0.0006900, 0.0097900, -0.0092571, 0.0088563
6: -0.0003485, 0.0127318, -0.0001365, 0.0126725, -0.0130209, 0.0128682
7: -0.0191234, 0.0004435, -0.0196524, 0.0001024, -0.0155988, 0.0167295
8: 0.9689999, 1.0250618, 0.9674844, 1.0240846, -0.0550846, 0.0575774
9: -0.0097126, 0.0067641, -0.0094253, 0.0072095, -0.0159402, 0.0150593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352629, upper bound: 0.0355833
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352629, upper bound: 0.0356058
time: 1.23 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002958, 0.0041676, -0.0002910, 0.0045291, -0.0043325, 0.0039856
1: -0.0000646, 0.0077086, -0.0000421, 0.0082627, -0.0083273, 0.0077507
2: 0.0047955, 0.0164368, 0.0039658, 0.0164031, -0.0116076, 0.0124710
3: -0.0070209, 0.0017329, -0.0076448, 0.0017075, -0.0087284, 0.0093777
4: -0.0108557, -0.0016425, -0.0114312, -0.0014187, -0.0094370, 0.0097887
5: 0.0009301, 0.0096679, 0.0003073, 0.0096426, -0.0087125, 0.0093606
6: 0.0001875, 0.0125819, -0.0006528, 0.0128169, -0.0126294, 0.0132347
7: -0.0193873, -0.0004188, -0.0193325, 0.0009333, -0.0167985, 0.0154425
8: 0.9682440, 1.0225914, 0.9684011, 1.0264651, -0.0582211, 0.0541903
9: -0.0089865, 0.0069863, -0.0101250, 0.0069401, -0.0147894, 0.0161518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351797, upper bound: 0.0355100
time: 0.96 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351751, upper bound: 0.0355064
time: 1.29 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002727, 0.0043981, -0.0002910, 0.0045291, -0.0042516, 0.0041569
1: 0.0000435, 0.0080620, -0.0000421, 0.0082627, -0.0082191, 0.0081041
2: 0.0042663, 0.0162748, 0.0039658, 0.0164031, -0.0121368, 0.0123091
3: -0.0074188, 0.0016111, -0.0076448, 0.0017075, -0.0091263, 0.0092559
4: -0.0112227, -0.0014998, -0.0114312, -0.0014187, -0.0098040, 0.0099314
5: 0.0005329, 0.0095463, 0.0003073, 0.0096426, -0.0091097, 0.0092390
6: -0.0003485, 0.0127318, -0.0006528, 0.0128169, -0.0131654, 0.0133846
7: -0.0191234, 0.0004435, -0.0193325, 0.0009333, -0.0158203, 0.0156712
8: 0.9689999, 1.0250618, 0.9684011, 1.0264651, -0.0574651, 0.0566608
9: -0.0097126, 0.0067641, -0.0101250, 0.0069401, -0.0151612, 0.0154503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351751, upper bound: 0.0354520
time: 1.10 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351751, upper bound: 0.0354738
time: 1.22 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0008122, 0.0043118, -0.0007814, 0.0042428, -0.0044625, 0.0045029
1: -0.0004813, 0.0079297, -0.0004674, 0.0078240, -0.0083053, 0.0083971
2: 0.0044644, 0.0179383, 0.0046227, 0.0178607, -0.0133963, 0.0133156
3: -0.0072699, 0.0024468, -0.0071508, 0.0024155, -0.0096853, 0.0095976
4: -0.0110854, -0.0013734, -0.0109755, -0.0014407, -0.0096447, 0.0096022
5: 0.0006723, 0.0100906, 0.0007915, 0.0100779, -0.0094057, 0.0092990
6: -0.0001479, 0.0126757, 0.0000125, 0.0126308, -0.0127787, 0.0126632
7: -0.0206528, 0.0001208, -0.0206032, -0.0001372, -0.0160813, 0.0163051
8: 0.9654623, 1.0241371, 0.9655509, 1.0233980, -0.0579358, 0.0585862
9: -0.0094678, 0.0076562, -0.0092492, 0.0076399, -0.0156684, 0.0154869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352952, upper bound: 0.0351830
time: 1.04 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352952, upper bound: 0.0351830
time: 1.00 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0007988, 0.0042827, -0.0105224, 0.0041435, -0.0043970, 0.0141868
1: -0.0004756, 0.0078851, -0.0026316, 0.0076718, -0.0081473, 0.0105167
2: 0.0045312, 0.0179049, 0.0048507, 0.0413613, -0.0358772, 0.0130542
3: -0.0072196, 0.0024335, -0.0069794, 0.0113036, -0.0176493, 0.0094129
4: -0.0110390, -0.0014025, -0.0108174, 0.0199953, -0.0310343, 0.0094149
5: 0.0007228, 0.0100855, 0.0007219, 0.0119273, -0.0108039, 0.0093636
6: -0.0000803, 0.0126567, 0.0002433, 0.0125662, -0.0126465, 0.0124134
7: -0.0206320, 0.0000120, -0.0339395, -0.0005086, -0.0161240, 0.0290162
8: 0.9654984, 1.0238254, 0.9512362, 1.0223339, -0.0568355, 0.0725892
9: -0.0093753, 0.0076499, -0.0097345, 0.0097071, -0.0164349, 0.0173844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
time: 1.05 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.43 seconds
IS_B1_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355233, upper bound: 0.0355390
IS_B1_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355233, upper bound: 0.0355694
IS_B1_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355233, upper bound: 0.0355390
IS_B1_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355233, upper bound: 0.0355694
IS_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0354733, upper bound: 0.0355719
IS_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0354982, upper bound: 0.0355719
IS_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0354982, upper bound: 0.0354733
IS_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0354982, upper bound: 0.0354982
IS_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355833, upper bound: 0.0352629
IS_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0356058, upper bound: 0.0352629
IS_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355833, upper bound: 0.0353016
IS_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0356058, upper bound: 0.0353016
IS_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355100, upper bound: 0.0351797
IS_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355064, upper bound: 0.0351751
IS_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0354937, upper bound: 0.0351751
IS_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0355064, upper bound: 0.0351751
IS_B2_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0352629, upper bound: 0.0355833
IS_B2_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0352629, upper bound: 0.0356058
IS_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0352629, upper bound: 0.0355833
IS_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0352629, upper bound: 0.0356058
IS_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0351797, upper bound: 0.0355100
IS_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0351751, upper bound: 0.0355064
IS_B2_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0351751, upper bound: 0.0354520
IS_B2_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0351751, upper bound: 0.0354738
IS_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0352952, upper bound: 0.0351830
IS_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0352952, upper bound: 0.0351830
IS_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
IS_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598

## BFS IS instance: IS_B1_A1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003176, 0.0041615, -0.0039355, 0.0038418
1: -0.0000590, 0.0075128, -0.0001666, 0.0076993, -0.0077583, 0.0076793
2: 0.0050888, 0.0164284, 0.0048095, 0.0165894, -0.0115006, 0.0116189
3: -0.0068003, 0.0017265, -0.0070104, 0.0018476, -0.0086480, 0.0087369
4: -0.0106523, -0.0017216, -0.0108460, -0.0016463, -0.0090060, 0.0091244
5: 0.0011502, 0.0096616, 0.0009406, 0.0097824, -0.0086322, 0.0087210
6: 0.0004845, 0.0124988, 0.0002016, 0.0125779, -0.0120935, 0.0122972
7: -0.0193736, -0.0008966, -0.0196361, -0.0004415, -0.0150622, 0.0149034
8: 0.9682832, 1.0212221, 0.9675312, 1.0225260, -0.0542428, 0.0536909
9: -0.0085841, 0.0069748, -0.0089673, 0.0071957, -0.0144928, 0.0146266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357502, upper bound: 0.0357502
time: 0.90 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357502, upper bound: 0.0357502
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003175, 0.0041566, -0.0039751, 0.0038603
1: -0.0001261, 0.0075326, -0.0001662, 0.0076918, -0.0078178, 0.0076988
2: 0.0050591, 0.0165288, 0.0048207, 0.0165889, -0.0115298, 0.0117081
3: -0.0068226, 0.0018021, -0.0070019, 0.0018473, -0.0086699, 0.0088040
4: -0.0106728, -0.0017136, -0.0108382, -0.0016493, -0.0090236, 0.0091246
5: 0.0011280, 0.0097369, 0.0009490, 0.0097821, -0.0086541, 0.0087879
6: 0.0004544, 0.0125072, 0.0002130, 0.0125747, -0.0121203, 0.0122942
7: -0.0195373, -0.0008483, -0.0196352, -0.0004598, -0.0155035, 0.0149487
8: 0.9678143, 1.0213606, 0.9675337, 1.0224736, -0.0546592, 0.0538269
9: -0.0086248, 0.0071126, -0.0089519, 0.0071950, -0.0145271, 0.0149061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357502, upper bound: 0.0357725
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0357502, upper bound: 0.0357725
time: 1.16 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0003176, 0.0041615, -0.0039442, 0.0041075
1: 0.0000497, 0.0078693, -0.0001666, 0.0076993, -0.0076496, 0.0080359
2: 0.0045548, 0.0162656, 0.0048095, 0.0165894, -0.0120346, 0.0114561
3: -0.0072019, 0.0016041, -0.0070104, 0.0018476, -0.0090495, 0.0086145
4: -0.0110227, -0.0015776, -0.0108460, -0.0016463, -0.0093764, 0.0092684
5: 0.0007494, 0.0095394, 0.0009406, 0.0097824, -0.0090330, 0.0085988
6: -0.0000563, 0.0126500, 0.0002016, 0.0125779, -0.0126342, 0.0124484
7: -0.0191083, -0.0000265, -0.0196361, -0.0004415, -0.0150158, 0.0162367
8: 0.9690432, 1.0237151, 0.9675312, 1.0225260, -0.0534828, 0.0561839
9: -0.0093168, 0.0067514, -0.0089673, 0.0071957, -0.0156155, 0.0144991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355660, upper bound: 0.0355378
time: 0.92 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355660, upper bound: 0.0355378
time: 1.04 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0003175, 0.0041566, -0.0039777, 0.0041184
1: -0.0000119, 0.0078797, -0.0001662, 0.0076918, -0.0077037, 0.0080459
2: 0.0045393, 0.0163578, 0.0048207, 0.0165889, -0.0120496, 0.0115371
3: -0.0072136, 0.0016735, -0.0070019, 0.0018473, -0.0090608, 0.0086754
4: -0.0110334, -0.0015734, -0.0108382, -0.0016493, -0.0093842, 0.0092648
5: 0.0007377, 0.0096086, 0.0009490, 0.0097821, -0.0090443, 0.0086596
6: -0.0000721, 0.0126545, 0.0002130, 0.0125747, -0.0126468, 0.0124415
7: -0.0192586, -0.0000012, -0.0196352, -0.0004598, -0.0154324, 0.0162776
8: 0.9686127, 1.0237877, 0.9675337, 1.0224736, -0.0538608, 0.0562540
9: -0.0093381, 0.0068779, -0.0089519, 0.0071950, -0.0156461, 0.0147440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355660, upper bound: 0.0355694
time: 1.23 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355660, upper bound: 0.0355694
time: 1.49 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0041598, -0.0002925, 0.0042740, -0.0040762, 0.0039803
1: -0.0000643, 0.0076967, -0.0000491, 0.0078717, -0.0079360, 0.0077458
2: 0.0048133, 0.0164362, 0.0045512, 0.0164135, -0.0116002, 0.0118850
3: -0.0070075, 0.0017325, -0.0072046, 0.0017153, -0.0087228, 0.0089370
4: -0.0108434, -0.0016473, -0.0110251, -0.0015766, -0.0092668, 0.0093778
5: 0.0009434, 0.0096675, 0.0007467, 0.0096504, -0.0087070, 0.0089207
6: 0.0002055, 0.0125768, -0.0000600, 0.0126511, -0.0124456, 0.0126368
7: -0.0193865, -0.0004477, -0.0193494, -0.0000207, -0.0158302, 0.0154504
8: 0.9682463, 1.0225083, 0.9683527, 1.0237318, -0.0554855, 0.0541556
9: -0.0089621, 0.0069856, -0.0093217, 0.0069543, -0.0147889, 0.0153326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355378, upper bound: 0.0355660
time: 0.98 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355378, upper bound: 0.0355721
time: 1.01 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0041549, -0.0003058, 0.0042808, -0.0040881, 0.0040092
1: -0.0000639, 0.0076892, -0.0001113, 0.0078822, -0.0079461, 0.0078005
2: 0.0048245, 0.0164357, 0.0045355, 0.0165066, -0.0116821, 0.0119002
3: -0.0069991, 0.0017321, -0.0072164, 0.0017854, -0.0087844, 0.0089484
4: -0.0108356, -0.0016503, -0.0110360, -0.0015724, -0.0092632, 0.0093857
5: 0.0009519, 0.0096671, 0.0007349, 0.0097203, -0.0087684, 0.0089321
6: 0.0002168, 0.0125737, -0.0000758, 0.0126555, -0.0124387, 0.0126495
7: -0.0193856, -0.0004660, -0.0195011, 0.0000049, -0.0158574, 0.0158109
8: 0.9682488, 1.0224559, 0.9679180, 1.0238050, -0.0555562, 0.0545380
9: -0.0089467, 0.0069848, -0.0093432, 0.0070821, -0.0150042, 0.0153495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355694, upper bound: 0.0355660
time: 1.08 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355694, upper bound: 0.0355721
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0002937, 0.0043920, -0.0041115, 0.0040227
1: 0.0000497, 0.0078693, -0.0000550, 0.0080526, -0.0080029, 0.0079243
2: 0.0045548, 0.0162656, 0.0042804, 0.0164223, -0.0118675, 0.0119852
3: -0.0072019, 0.0016041, -0.0074082, 0.0017220, -0.0089238, 0.0090123
4: -0.0110227, -0.0015776, -0.0112130, -0.0015036, -0.0095191, 0.0096354
5: 0.0007494, 0.0095394, 0.0005434, 0.0096570, -0.0089076, 0.0089959
6: -0.0000563, 0.0126500, -0.0003342, 0.0127278, -0.0127841, 0.0129843
7: -0.0191083, -0.0000265, -0.0193637, 0.0004206, -0.0152705, 0.0151177
8: 0.9690432, 1.0237151, 0.9683115, 1.0249962, -0.0559530, 0.0554036
9: -0.0093168, 0.0067514, -0.0096933, 0.0069664, -0.0147777, 0.0149242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355647, upper bound: 0.0354711
time: 1.25 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355647, upper bound: 0.0354711
time: 1.41 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0002937, 0.0043870, -0.0041503, 0.0040315
1: -0.0000119, 0.0078797, -0.0000545, 0.0080449, -0.0080568, 0.0079343
2: 0.0045393, 0.0163578, 0.0042918, 0.0164217, -0.0118824, 0.0120660
3: -0.0072136, 0.0016735, -0.0073996, 0.0017215, -0.0089351, 0.0090731
4: -0.0110334, -0.0015734, -0.0112051, -0.0015066, -0.0095268, 0.0096317
5: 0.0007377, 0.0096086, 0.0005520, 0.0096565, -0.0089188, 0.0090566
6: -0.0000721, 0.0126545, -0.0003226, 0.0127245, -0.0127966, 0.0129771
7: -0.0192586, -0.0000012, -0.0193627, 0.0004020, -0.0157075, 0.0151343
8: 0.9686127, 1.0237877, 0.9683143, 1.0249428, -0.0563300, 0.0554734
9: -0.0093381, 0.0068779, -0.0096776, 0.0069656, -0.0147877, 0.0151918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355647, upper bound: 0.0354982
time: 0.99 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355647, upper bound: 0.0354982
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0003189, 0.0042992, -0.0002946, 0.0040398, -0.0038506, 0.0040832
1: -0.0001729, 0.0079104, -0.0000590, 0.0075128, -0.0076857, 0.0079694
2: 0.0044933, 0.0165989, 0.0050888, 0.0164284, -0.0119350, 0.0115101
3: -0.0072481, 0.0018548, -0.0068003, 0.0017265, -0.0089747, 0.0086551
4: -0.0110653, -0.0015610, -0.0106523, -0.0017216, -0.0093437, 0.0090913
5: 0.0007033, 0.0097896, 0.0011502, 0.0096616, -0.0089583, 0.0086393
6: -0.0001186, 0.0126675, 0.0004845, 0.0124988, -0.0126174, 0.0121830
7: -0.0196515, 0.0000737, -0.0193736, -0.0008966, -0.0149077, 0.0155751
8: 0.9674870, 1.0240021, 0.9682832, 1.0212221, -0.0537351, 0.0557189
9: -0.0094011, 0.0072087, -0.0085841, 0.0069748, -0.0150586, 0.0144014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0358619, upper bound: 0.0355478
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0358619, upper bound: 0.0355671
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0003189, 0.0042943, -0.0003089, 0.0040527, -0.0038691, 0.0041224
1: -0.0001725, 0.0079028, -0.0001261, 0.0075326, -0.0077051, 0.0080288
2: 0.0045047, 0.0165983, 0.0050591, 0.0165288, -0.0120241, 0.0115392
3: -0.0072395, 0.0018543, -0.0068226, 0.0018021, -0.0090416, 0.0086770
4: -0.0110574, -0.0015641, -0.0106728, -0.0017136, -0.0093438, 0.0091088
5: 0.0007118, 0.0097891, 0.0011280, 0.0097369, -0.0090251, 0.0086612
6: -0.0001070, 0.0126642, 0.0004544, 0.0125072, -0.0126142, 0.0122098
7: -0.0196505, 0.0000551, -0.0195373, -0.0008483, -0.0149527, 0.0160127
8: 0.9674898, 1.0239488, 0.9678143, 1.0213606, -0.0538709, 0.0561345
9: -0.0093855, 0.0072079, -0.0086248, 0.0071126, -0.0153349, 0.0144362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0358738, upper bound: 0.0355478
time: 1.00 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0358738, upper bound: 0.0355671
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0003189, 0.0042992, -0.0002714, 0.0042724, -0.0041164, 0.0040919
1: -0.0001729, 0.0079104, 0.0000497, 0.0078693, -0.0080422, 0.0078607
2: 0.0044933, 0.0165989, 0.0045548, 0.0162656, -0.0117722, 0.0120441
3: -0.0072481, 0.0018548, -0.0072019, 0.0016041, -0.0088522, 0.0090567
4: -0.0110653, -0.0015610, -0.0110227, -0.0015776, -0.0094877, 0.0094617
5: 0.0007033, 0.0097896, 0.0007494, 0.0095394, -0.0088361, 0.0090401
6: -0.0001186, 0.0126675, -0.0000563, 0.0126500, -0.0127686, 0.0127238
7: -0.0196515, 0.0000737, -0.0191083, -0.0000265, -0.0162410, 0.0155287
8: 0.9674870, 1.0240021, 0.9690432, 1.0237151, -0.0562281, 0.0549589
9: -0.0094011, 0.0072087, -0.0093168, 0.0067514, -0.0149311, 0.0155241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355827, upper bound: 0.0352905
time: 1.26 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355827, upper bound: 0.0353016
time: 1.11 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0003189, 0.0042943, -0.0002845, 0.0042792, -0.0041273, 0.0041250
1: -0.0001725, 0.0079028, -0.0000119, 0.0078797, -0.0080522, 0.0079147
2: 0.0045047, 0.0165983, 0.0045393, 0.0163578, -0.0118530, 0.0120590
3: -0.0072395, 0.0018543, -0.0072136, 0.0016735, -0.0089130, 0.0090679
4: -0.0110574, -0.0015641, -0.0110334, -0.0015734, -0.0094840, 0.0094694
5: 0.0007118, 0.0097891, 0.0007377, 0.0096086, -0.0088968, 0.0090514
6: -0.0001070, 0.0126642, -0.0000721, 0.0126545, -0.0127615, 0.0127363
7: -0.0196505, 0.0000551, -0.0192586, -0.0000012, -0.0162816, 0.0159416
8: 0.9674898, 1.0239488, 0.9686127, 1.0237877, -0.0562980, 0.0553361
9: -0.0093855, 0.0072079, -0.0093381, 0.0068779, -0.0151727, 0.0155552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356058, upper bound: 0.0352905
time: 0.96 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0356058, upper bound: 0.0353016
time: 1.01 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002896, 0.0043996, -0.0002957, 0.0041598, -0.0039757, 0.0042011
1: -0.0000357, 0.0080642, -0.0000643, 0.0076967, -0.0077325, 0.0081285
2: 0.0042630, 0.0163935, 0.0048133, 0.0164362, -0.0121732, 0.0115802
3: -0.0074213, 0.0017003, -0.0070075, 0.0017325, -0.0091538, 0.0087078
4: -0.0112251, -0.0014989, -0.0108434, -0.0016473, -0.0095778, 0.0093445
5: 0.0005304, 0.0096354, 0.0009434, 0.0096675, -0.0091371, 0.0086919
6: -0.0003518, 0.0127327, 0.0002055, 0.0125768, -0.0129287, 0.0125272
7: -0.0193168, 0.0004490, -0.0193865, -0.0004477, -0.0153697, 0.0162817
8: 0.9684460, 1.0250775, 0.9682463, 1.0225083, -0.0540622, 0.0568311
9: -0.0097171, 0.0069269, -0.0089621, 0.0069856, -0.0157128, 0.0146477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354973, upper bound: 0.0352234
time: 0.99 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354973, upper bound: 0.0352234
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003028, 0.0044103, -0.0002957, 0.0041549, -0.0040036, 0.0042130
1: -0.0000973, 0.0080806, -0.0000639, 0.0076892, -0.0077865, 0.0081445
2: 0.0042384, 0.0164857, 0.0048245, 0.0164357, -0.0121973, 0.0116611
3: -0.0074398, 0.0017696, -0.0069991, 0.0017321, -0.0091718, 0.0087687
4: -0.0112421, -0.0014922, -0.0108356, -0.0016503, -0.0095918, 0.0093433
5: 0.0005120, 0.0097046, 0.0009519, 0.0096671, -0.0091551, 0.0087527
6: -0.0003767, 0.0127397, 0.0002168, 0.0125737, -0.0129504, 0.0125228
7: -0.0194670, 0.0004890, -0.0193856, -0.0004660, -0.0157259, 0.0163099
8: 0.9680156, 1.0251920, 0.9682488, 1.0224559, -0.0544403, 0.0569432
9: -0.0097508, 0.0070534, -0.0089467, 0.0069848, -0.0157305, 0.0148775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354973, upper bound: 0.0352555
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354973, upper bound: 0.0352555
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0002909, 0.0045214, -0.0002714, 0.0042724, -0.0040283, 0.0042423
1: -0.0000417, 0.0082509, 0.0000497, 0.0078693, -0.0079111, 0.0082012
2: 0.0039834, 0.0164025, 0.0045548, 0.0162656, -0.0122822, 0.0118477
3: -0.0076315, 0.0017071, -0.0072019, 0.0016041, -0.0092357, 0.0089090
4: -0.0114190, -0.0014235, -0.0110227, -0.0015776, -0.0098414, 0.0095992
5: 0.0003205, 0.0096421, 0.0007494, 0.0095394, -0.0092188, 0.0088927
6: -0.0006350, 0.0128119, -0.0000563, 0.0126500, -0.0132850, 0.0128682
7: -0.0193315, 0.0009046, -0.0191083, -0.0000265, -0.0151092, 0.0157505
8: 0.9684039, 1.0263827, 0.9690432, 1.0237151, -0.0553113, 0.0573395
9: -0.0101008, 0.0069393, -0.0093168, 0.0067514, -0.0153284, 0.0146834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354915, upper bound: 0.0351617
time: 1.06 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0354915, upper bound: 0.0351751
time: 1.02 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0002908, 0.0045163, -0.0002845, 0.0042792, -0.0040370, 0.0042804
1: -0.0000413, 0.0082431, -0.0000119, 0.0078797, -0.0079210, 0.0082550
2: 0.0039950, 0.0164019, 0.0045393, 0.0163578, -0.0123628, 0.0118626
3: -0.0076228, 0.0017066, -0.0072136, 0.0016735, -0.0092963, 0.0089202
4: -0.0114109, -0.0014266, -0.0110334, -0.0015734, -0.0098376, 0.0096069
5: 0.0003292, 0.0096417, 0.0007377, 0.0096086, -0.0092794, 0.0089039
6: -0.0006232, 0.0128086, -0.0000721, 0.0126545, -0.0132777, 0.0128807
7: -0.0193305, 0.0008857, -0.0192586, -0.0000012, -0.0151255, 0.0161849
8: 0.9684068, 1.0263286, 0.9686127, 1.0237877, -0.0553809, 0.0577158
9: -0.0100849, 0.0069384, -0.0093381, 0.0068779, -0.0155937, 0.0146935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355064, upper bound: 0.0351618
time: 1.56 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355064, upper bound: 0.0351751
time: 0.86 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003189, 0.0042992, -0.0040832, 0.0038506
1: -0.0000590, 0.0075128, -0.0001729, 0.0079104, -0.0079694, 0.0076857
2: 0.0050888, 0.0164284, 0.0044933, 0.0165989, -0.0115101, 0.0119350
3: -0.0068003, 0.0017265, -0.0072481, 0.0018548, -0.0086551, 0.0089747
4: -0.0106523, -0.0017216, -0.0110653, -0.0015610, -0.0090913, 0.0093437
5: 0.0011502, 0.0096616, 0.0007033, 0.0097896, -0.0086393, 0.0089583
6: 0.0004845, 0.0124988, -0.0001186, 0.0126675, -0.0121830, 0.0126174
7: -0.0193736, -0.0008966, -0.0196515, 0.0000737, -0.0155751, 0.0149077
8: 0.9682832, 1.0212221, 0.9674870, 1.0240021, -0.0557189, 0.0537351
9: -0.0085841, 0.0069748, -0.0094011, 0.0072087, -0.0144014, 0.0150586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355478, upper bound: 0.0358619
time: 0.92 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355478, upper bound: 0.0358619
time: 1.09 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003189, 0.0042943, -0.0041224, 0.0038691
1: -0.0001261, 0.0075326, -0.0001725, 0.0079028, -0.0080288, 0.0077051
2: 0.0050591, 0.0165288, 0.0045047, 0.0165983, -0.0115392, 0.0120241
3: -0.0068226, 0.0018021, -0.0072395, 0.0018543, -0.0086770, 0.0090416
4: -0.0106728, -0.0017136, -0.0110574, -0.0015641, -0.0091088, 0.0093438
5: 0.0011280, 0.0097369, 0.0007118, 0.0097891, -0.0086612, 0.0090251
6: 0.0004544, 0.0125072, -0.0001070, 0.0126642, -0.0122098, 0.0126142
7: -0.0195373, -0.0008483, -0.0196505, 0.0000551, -0.0160127, 0.0149528
8: 0.9678143, 1.0213606, 0.9674898, 1.0239488, -0.0561345, 0.0538709
9: -0.0086248, 0.0071126, -0.0093855, 0.0072079, -0.0144362, 0.0153349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355478, upper bound: 0.0358738
time: 1.08 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0355478, upper bound: 0.0358738
time: 1.14 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0003189, 0.0042992, -0.0040919, 0.0041164
1: 0.0000497, 0.0078693, -0.0001729, 0.0079104, -0.0078607, 0.0080422
2: 0.0045548, 0.0162656, 0.0044933, 0.0165989, -0.0120441, 0.0117722
3: -0.0072019, 0.0016041, -0.0072481, 0.0018548, -0.0090567, 0.0088522
4: -0.0110227, -0.0015776, -0.0110653, -0.0015610, -0.0094617, 0.0094877
5: 0.0007494, 0.0095394, 0.0007033, 0.0097896, -0.0090401, 0.0088361
6: -0.0000563, 0.0126500, -0.0001186, 0.0126675, -0.0127238, 0.0127686
7: -0.0191083, -0.0000265, -0.0196515, 0.0000737, -0.0155287, 0.0162410
8: 0.9690432, 1.0237151, 0.9674870, 1.0240021, -0.0549589, 0.0562281
9: -0.0093168, 0.0067514, -0.0094011, 0.0072087, -0.0155241, 0.0149311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352905, upper bound: 0.0355827
time: 1.32 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352905, upper bound: 0.0355827
time: 1.28 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0003189, 0.0042943, -0.0041250, 0.0041273
1: -0.0000119, 0.0078797, -0.0001725, 0.0079028, -0.0079147, 0.0080522
2: 0.0045393, 0.0163578, 0.0045047, 0.0165983, -0.0120590, 0.0118530
3: -0.0072136, 0.0016735, -0.0072395, 0.0018543, -0.0090679, 0.0089130
4: -0.0110334, -0.0015734, -0.0110574, -0.0015641, -0.0094694, 0.0094840
5: 0.0007377, 0.0096086, 0.0007118, 0.0097891, -0.0090514, 0.0088968
6: -0.0000721, 0.0126545, -0.0001070, 0.0126642, -0.0127363, 0.0127615
7: -0.0192586, -0.0000012, -0.0196505, 0.0000551, -0.0159416, 0.0162816
8: 0.9686127, 1.0237877, 0.9674898, 1.0239488, -0.0553361, 0.0562980
9: -0.0093381, 0.0068779, -0.0093855, 0.0072079, -0.0155552, 0.0151727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352905, upper bound: 0.0356058
time: 0.98 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352905, upper bound: 0.0356058
time: 1.07 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0041598, -0.0002896, 0.0043996, -0.0042011, 0.0039757
1: -0.0000643, 0.0076967, -0.0000357, 0.0080642, -0.0081285, 0.0077325
2: 0.0048133, 0.0164362, 0.0042630, 0.0163935, -0.0115802, 0.0121732
3: -0.0070075, 0.0017325, -0.0074213, 0.0017003, -0.0087078, 0.0091538
4: -0.0108434, -0.0016473, -0.0112251, -0.0014989, -0.0093445, 0.0095778
5: 0.0009434, 0.0096675, 0.0005304, 0.0096354, -0.0086919, 0.0091371
6: 0.0002055, 0.0125768, -0.0003518, 0.0127327, -0.0125272, 0.0129287
7: -0.0193865, -0.0004477, -0.0193168, 0.0004490, -0.0162817, 0.0153697
8: 0.9682463, 1.0225083, 0.9684460, 1.0250775, -0.0568311, 0.0540622
9: -0.0089621, 0.0069856, -0.0097171, 0.0069269, -0.0146476, 0.0157128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352234, upper bound: 0.0354973
time: 1.05 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352234, upper bound: 0.0355090
time: 1.02 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002957, 0.0041549, -0.0003028, 0.0044103, -0.0042130, 0.0040036
1: -0.0000639, 0.0076892, -0.0000973, 0.0080806, -0.0081445, 0.0077865
2: 0.0048245, 0.0164357, 0.0042384, 0.0164857, -0.0116611, 0.0121973
3: -0.0069991, 0.0017321, -0.0074398, 0.0017696, -0.0087687, 0.0091718
4: -0.0108356, -0.0016503, -0.0112421, -0.0014922, -0.0093433, 0.0095918
5: 0.0009519, 0.0096671, 0.0005120, 0.0097046, -0.0087527, 0.0091551
6: 0.0002168, 0.0125737, -0.0003767, 0.0127397, -0.0125228, 0.0129504
7: -0.0193856, -0.0004660, -0.0194670, 0.0004890, -0.0163099, 0.0157259
8: 0.9682488, 1.0224559, 0.9680156, 1.0251920, -0.0569432, 0.0544403
9: -0.0089467, 0.0069848, -0.0097508, 0.0070534, -0.0148775, 0.0157305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352555, upper bound: 0.0354973
time: 1.22 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352555, upper bound: 0.0355090
time: 1.33 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0002909, 0.0045214, -0.0042423, 0.0040283
1: 0.0000497, 0.0078693, -0.0000417, 0.0082509, -0.0082012, 0.0079111
2: 0.0045548, 0.0162656, 0.0039834, 0.0164025, -0.0118477, 0.0122822
3: -0.0072019, 0.0016041, -0.0076315, 0.0017071, -0.0089090, 0.0092357
4: -0.0110227, -0.0015776, -0.0114190, -0.0014235, -0.0095992, 0.0098414
5: 0.0007494, 0.0095394, 0.0003205, 0.0096421, -0.0088927, 0.0092188
6: -0.0000563, 0.0126500, -0.0006350, 0.0128119, -0.0128682, 0.0132850
7: -0.0191083, -0.0000265, -0.0193315, 0.0009046, -0.0157505, 0.0151092
8: 0.9690432, 1.0237151, 0.9684039, 1.0263827, -0.0573395, 0.0553113
9: -0.0093168, 0.0067514, -0.0101008, 0.0069393, -0.0146834, 0.0153284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352712, upper bound: 0.0354519
time: 1.29 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352712, upper bound: 0.0354519
time: 1.38 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0002908, 0.0045163, -0.0042804, 0.0040370
1: -0.0000119, 0.0078797, -0.0000413, 0.0082431, -0.0082550, 0.0079210
2: 0.0045393, 0.0163578, 0.0039950, 0.0164019, -0.0118626, 0.0123628
3: -0.0072136, 0.0016735, -0.0076228, 0.0017066, -0.0089202, 0.0092963
4: -0.0110334, -0.0015734, -0.0114109, -0.0014266, -0.0096069, 0.0098376
5: 0.0007377, 0.0096086, 0.0003292, 0.0096417, -0.0089039, 0.0092794
6: -0.0000721, 0.0126545, -0.0006232, 0.0128086, -0.0128807, 0.0132777
7: -0.0192586, -0.0000012, -0.0193305, 0.0008857, -0.0161849, 0.0151255
8: 0.9686127, 1.0237877, 0.9684068, 1.0263286, -0.0577158, 0.0553809
9: -0.0093381, 0.0068779, -0.0100849, 0.0069384, -0.0146935, 0.0155937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352712, upper bound: 0.0354738
time: 0.98 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352712, upper bound: 0.0354738
time: 0.86 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0007814, 0.0042428, -0.0007814, 0.0042428, -0.0044332, 0.0044332
1: -0.0004674, 0.0078240, -0.0004674, 0.0078240, -0.0082913, 0.0082913
2: 0.0046227, 0.0178607, 0.0046227, 0.0178607, -0.0132379, 0.0132379
3: -0.0071508, 0.0024155, -0.0071508, 0.0024155, -0.0095663, 0.0095663
4: -0.0109755, -0.0014407, -0.0109755, -0.0014407, -0.0095349, 0.0095349
5: 0.0007915, 0.0100779, 0.0007915, 0.0100779, -0.0092864, 0.0092864
6: 0.0000125, 0.0126308, 0.0000125, 0.0126308, -0.0126184, 0.0126184
7: -0.0206032, -0.0001372, -0.0206032, -0.0001372, -0.0160384, 0.0160384
8: 0.9655509, 1.0233980, 0.9655509, 1.0233980, -0.0578471, 0.0578471
9: -0.0092492, 0.0076399, -0.0092492, 0.0076399, -0.0154340, 0.0154340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352535, upper bound: 0.0355870
time: 0.93 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352477, upper bound: 0.0355870
time: 0.88 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0105224, 0.0041435, -0.0007814, 0.0042428, -0.0141457, 0.0043476
1: -0.0026316, 0.0076718, -0.0004674, 0.0078240, -0.0104556, 0.0081391
2: 0.0048507, 0.0413613, 0.0046227, 0.0178607, -0.0130100, 0.0357748
3: -0.0069794, 0.0113036, -0.0071508, 0.0024155, -0.0093949, 0.0175723
4: -0.0108174, 0.0199953, -0.0109755, -0.0014407, -0.0093767, 0.0309708
5: 0.0007219, 0.0119273, 0.0007915, 0.0100779, -0.0093560, 0.0107258
6: 0.0002433, 0.0125662, 0.0000125, 0.0126308, -0.0123875, 0.0125538
7: -0.0339395, -0.0005086, -0.0206032, -0.0001372, -0.0288494, 0.0158194
8: 0.9512362, 1.0223339, 0.9655509, 1.0233980, -0.0721618, 0.0567830
9: -0.0097345, 0.0097071, -0.0092492, 0.0076399, -0.0173744, 0.0162895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352535, upper bound: 0.0355870
time: 0.98 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0352477, upper bound: 0.0355870
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005951, 0.0041561, -0.0105105, 0.0041356, -0.0042014, 0.0140395
1: -0.0004044, 0.0076911, -0.0026277, 0.0076596, -0.0080640, 0.0103188
2: 0.0048218, 0.0174112, 0.0048689, 0.0413323, -0.0354863, 0.0125423
3: -0.0070011, 0.0022452, -0.0069657, 0.0112925, -0.0173664, 0.0092109
4: -0.0108375, -0.0016155, -0.0108048, 0.0199687, -0.0308062, 0.0091893
5: 0.0009445, 0.0100254, 0.0007352, 0.0119240, -0.0105081, 0.0092902
6: 0.0002140, 0.0125744, 0.0002617, 0.0125611, -0.0123471, 0.0123127
7: -0.0203477, -0.0004615, -0.0339228, -0.0005383, -0.0158987, 0.0284156
8: 0.9659388, 1.0224688, 0.9512550, 1.0222487, -0.0563099, 0.0712138
9: -0.0089658, 0.0075849, -0.0097101, 0.0097036, -0.0158768, 0.0172950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
time: 0.94 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
time: 0.98 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0019884, 0.0041740, -0.0104949, 0.0041313, -0.0055650, 0.0140475
1: -0.0008714, 0.0077185, -0.0026243, 0.0076531, -0.0085245, 0.0103428
2: 0.0047807, 0.0207959, 0.0048787, 0.0412945, -0.0354791, 0.0154913
3: -0.0070320, 0.0035421, -0.0069584, 0.0112781, -0.0173755, 0.0105005
4: -0.0108660, 0.0011854, -0.0107980, 0.0199342, -0.0308002, 0.0119835
5: 0.0008859, 0.0104268, 0.0007436, 0.0119211, -0.0106932, 0.0096832
6: 0.0001724, 0.0125861, 0.0002716, 0.0125583, -0.0123859, 0.0123144
7: -0.0223186, -0.0003945, -0.0339011, -0.0005542, -0.0177115, 0.0284490
8: 0.9630294, 1.0226607, 0.9512785, 1.0222032, -0.0591738, 0.0713822
9: -0.0091030, 0.0080403, -0.0096956, 0.0097003, -0.0164446, 0.0177359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598
time: 1.00 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598
time: 0.98 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.61 seconds
IS_B1_A1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0357502, upper bound: 0.0357502
IS_B1_A1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0357502, upper bound: 0.0357502
IS_B1_A1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0357502, upper bound: 0.0357725
IS_B1_A1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0357502, upper bound: 0.0357725
IS_B1_A1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355660, upper bound: 0.0355378
IS_B1_A1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355660, upper bound: 0.0355378
IS_B1_A1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355660, upper bound: 0.0355694
IS_B1_A1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355660, upper bound: 0.0355694
IS_B1_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355378, upper bound: 0.0355660
IS_B1_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355378, upper bound: 0.0355721
IS_B1_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355694, upper bound: 0.0355660
IS_B1_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355694, upper bound: 0.0355721
IS_B1_A1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355647, upper bound: 0.0354711
IS_B1_A1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355647, upper bound: 0.0354711
IS_B1_A1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355647, upper bound: 0.0354982
IS_B1_A1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355647, upper bound: 0.0354982
IS_B1_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0358619, upper bound: 0.0355478
IS_B1_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0358619, upper bound: 0.0355671
IS_B1_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0358738, upper bound: 0.0355478
IS_B1_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0358738, upper bound: 0.0355671
IS_B1_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355827, upper bound: 0.0352905
IS_B1_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355827, upper bound: 0.0353016
IS_B1_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0356058, upper bound: 0.0352905
IS_B1_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0356058, upper bound: 0.0353016
IS_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0354973, upper bound: 0.0352234
IS_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0354973, upper bound: 0.0352234
IS_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0354973, upper bound: 0.0352555
IS_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0354973, upper bound: 0.0352555
IS_B1_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0354915, upper bound: 0.0351617
IS_B1_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0354915, upper bound: 0.0351751
IS_B1_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355064, upper bound: 0.0351618
IS_B1_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355064, upper bound: 0.0351751
IS_B2_A2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355478, upper bound: 0.0358619
IS_B2_A2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355478, upper bound: 0.0358619
IS_B2_A2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355478, upper bound: 0.0358738
IS_B2_A2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0355478, upper bound: 0.0358738
IS_B2_A2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352905, upper bound: 0.0355827
IS_B2_A2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352905, upper bound: 0.0355827
IS_B2_A2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352905, upper bound: 0.0356058
IS_B2_A2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352905, upper bound: 0.0356058
IS_B2_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352234, upper bound: 0.0354973
IS_B2_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352234, upper bound: 0.0355090
IS_B2_A2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352555, upper bound: 0.0354973
IS_B2_A2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352555, upper bound: 0.0355090
IS_B2_A2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352712, upper bound: 0.0354519
IS_B2_A2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352712, upper bound: 0.0354519
IS_B2_A2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352712, upper bound: 0.0354738
IS_B2_A2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352712, upper bound: 0.0354738
IS_B2_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352535, upper bound: 0.0355870
IS_B2_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352477, upper bound: 0.0355870
IS_B2_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352535, upper bound: 0.0355870
IS_B2_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0352477, upper bound: 0.0355870
IS_B2_A2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
IS_B2_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
IS_B2_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598
IS_B2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.61
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598

## BFS IS instance: IS_B1_A1_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003165, 0.0040414, -0.0038106, 0.0038406
1: -0.0000590, 0.0075128, -0.0001614, 0.0075153, -0.0075743, 0.0076742
2: 0.0050888, 0.0164284, 0.0050850, 0.0165817, -0.0114929, 0.0113433
3: -0.0068003, 0.0017265, -0.0068032, 0.0018419, -0.0086422, 0.0085297
4: -0.0106523, -0.0017216, -0.0106549, -0.0017206, -0.0089317, 0.0089333
5: 0.0011502, 0.0096616, 0.0011474, 0.0097767, -0.0086264, 0.0085142
6: 0.0004845, 0.0124988, 0.0004806, 0.0124999, -0.0120154, 0.0120182
7: -0.0193736, -0.0008966, -0.0196236, -0.0008905, -0.0145132, 0.0148713
8: 0.9682832, 1.0212221, 0.9675671, 1.0212396, -0.0529565, 0.0536550
9: -0.0085841, 0.0069748, -0.0085892, 0.0071852, -0.0143950, 0.0141643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347265, upper bound: 0.0341323
time: 0.83 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340202, upper bound: 0.0340208
time: 0.95 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003310, 0.0040544, -0.0038446, 0.0038819
1: -0.0000590, 0.0075128, -0.0002295, 0.0075351, -0.0075941, 0.0077423
2: 0.0050888, 0.0164284, 0.0050553, 0.0166837, -0.0115949, 0.0113730
3: -0.0068003, 0.0017265, -0.0068255, 0.0019185, -0.0087188, 0.0085521
4: -0.0106523, -0.0017216, -0.0106755, -0.0017126, -0.0089397, 0.0089539
5: 0.0011502, 0.0096616, 0.0011251, 0.0098532, -0.0087030, 0.0085365
6: 0.0004845, 0.0124988, 0.0004505, 0.0125083, -0.0120238, 0.0120483
7: -0.0193736, -0.0008966, -0.0197897, -0.0008421, -0.0148690, 0.0152783
8: 0.9682832, 1.0212221, 0.9670913, 1.0213785, -0.0530953, 0.0541308
9: -0.0085841, 0.0069748, -0.0086300, 0.0073251, -0.0146310, 0.0144639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347265, upper bound: 0.0341323
time: 1.05 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340202, upper bound: 0.0340208
time: 1.00 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003165, 0.0040414, -0.0038528, 0.0038724
1: -0.0001261, 0.0075326, -0.0001614, 0.0075153, -0.0076414, 0.0076940
2: 0.0050591, 0.0165288, 0.0050850, 0.0165817, -0.0115226, 0.0114438
3: -0.0068226, 0.0018021, -0.0068032, 0.0018419, -0.0086645, 0.0086052
4: -0.0106728, -0.0017136, -0.0106549, -0.0017206, -0.0089523, 0.0089413
5: 0.0011280, 0.0097369, 0.0011474, 0.0097767, -0.0086487, 0.0085895
6: 0.0004544, 0.0125072, 0.0004806, 0.0124999, -0.0120455, 0.0120266
7: -0.0195373, -0.0008483, -0.0196236, -0.0008905, -0.0149453, 0.0151815
8: 0.9678143, 1.0213606, 0.9675671, 1.0212396, -0.0534253, 0.0537935
9: -0.0086248, 0.0071126, -0.0085892, 0.0071852, -0.0146562, 0.0144360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347262, upper bound: 0.0341326
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340157, upper bound: 0.0340200
time: 0.78 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003310, 0.0040544, -0.0038306, 0.0038610
1: -0.0001261, 0.0075326, -0.0002295, 0.0075351, -0.0076612, 0.0077621
2: 0.0050591, 0.0165288, 0.0050553, 0.0166837, -0.0116245, 0.0114735
3: -0.0068226, 0.0018021, -0.0068255, 0.0019185, -0.0087412, 0.0086276
4: -0.0106728, -0.0017136, -0.0106755, -0.0017126, -0.0089603, 0.0089619
5: 0.0011280, 0.0097369, 0.0011251, 0.0098532, -0.0087252, 0.0086118
6: 0.0004544, 0.0125072, 0.0004505, 0.0125083, -0.0120539, 0.0120567
7: -0.0195373, -0.0008483, -0.0197897, -0.0008421, -0.0145629, 0.0149358
8: 0.9678143, 1.0213606, 0.9670913, 1.0213785, -0.0535642, 0.0542693
9: -0.0086248, 0.0071126, -0.0086300, 0.0073251, -0.0144464, 0.0141824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0347262, upper bound: 0.0341326
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340157, upper bound: 0.0340200
time: 0.95 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0003165, 0.0040414, -0.0038194, 0.0041063
1: 0.0000497, 0.0078693, -0.0001614, 0.0075153, -0.0074656, 0.0080308
2: 0.0045548, 0.0162656, 0.0050850, 0.0165817, -0.0120269, 0.0111805
3: -0.0072019, 0.0016041, -0.0068032, 0.0018419, -0.0090437, 0.0084073
4: -0.0110227, -0.0015776, -0.0106549, -0.0017206, -0.0093021, 0.0090773
5: 0.0007494, 0.0095394, 0.0011474, 0.0097767, -0.0090273, 0.0083920
6: -0.0000563, 0.0126500, 0.0004806, 0.0124999, -0.0125562, 0.0121694
7: -0.0191083, -0.0000265, -0.0196236, -0.0008905, -0.0144668, 0.0162046
8: 0.9690432, 1.0237151, 0.9675671, 1.0212396, -0.0521964, 0.0561480
9: -0.0093168, 0.0067514, -0.0085892, 0.0071852, -0.0155177, 0.0140369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344685, upper bound: 0.0338627
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335303, upper bound: 0.0336872
time: 1.49 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0003310, 0.0040544, -0.0038534, 0.0041477
1: 0.0000497, 0.0078693, -0.0002295, 0.0075351, -0.0074854, 0.0080988
2: 0.0045548, 0.0162656, 0.0050553, 0.0166837, -0.0121289, 0.0112102
3: -0.0072019, 0.0016041, -0.0068255, 0.0019185, -0.0091204, 0.0084296
4: -0.0110227, -0.0015776, -0.0106755, -0.0017126, -0.0093101, 0.0090979
5: 0.0007494, 0.0095394, 0.0011251, 0.0098532, -0.0091038, 0.0084143
6: -0.0000563, 0.0126500, 0.0004505, 0.0125083, -0.0125646, 0.0121995
7: -0.0191083, -0.0000265, -0.0197897, -0.0008421, -0.0148226, 0.0166117
8: 0.9690432, 1.0237151, 0.9670913, 1.0213785, -0.0523353, 0.0566238
9: -0.0093168, 0.0067514, -0.0086300, 0.0073251, -0.0157537, 0.0143365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344685, upper bound: 0.0338628
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335303, upper bound: 0.0336872
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0003165, 0.0040414, -0.0038554, 0.0041245
1: -0.0000119, 0.0078797, -0.0001614, 0.0075153, -0.0075272, 0.0080411
2: 0.0045393, 0.0163578, 0.0050850, 0.0165817, -0.0120425, 0.0112727
3: -0.0072136, 0.0016735, -0.0068032, 0.0018419, -0.0090554, 0.0084766
4: -0.0110334, -0.0015734, -0.0106549, -0.0017206, -0.0093129, 0.0090815
5: 0.0007377, 0.0096086, 0.0011474, 0.0097767, -0.0090389, 0.0084612
6: -0.0000721, 0.0126545, 0.0004806, 0.0124999, -0.0125720, 0.0121738
7: -0.0192586, -0.0000012, -0.0196236, -0.0008905, -0.0148742, 0.0163776
8: 0.9686127, 1.0237877, 0.9675671, 1.0212396, -0.0526269, 0.0562206
9: -0.0093381, 0.0068779, -0.0085892, 0.0071852, -0.0156634, 0.0142739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344685, upper bound: 0.0338700
time: 1.05 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335095, upper bound: 0.0337021
time: 0.95 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0003310, 0.0040544, -0.0038391, 0.0041191
1: -0.0000119, 0.0078797, -0.0002295, 0.0075351, -0.0075470, 0.0081092
2: 0.0045393, 0.0163578, 0.0050553, 0.0166837, -0.0121444, 0.0113025
3: -0.0072136, 0.0016735, -0.0068255, 0.0019185, -0.0091321, 0.0084990
4: -0.0110334, -0.0015734, -0.0106755, -0.0017126, -0.0093209, 0.0091021
5: 0.0007377, 0.0096086, 0.0011251, 0.0098532, -0.0091155, 0.0084835
6: -0.0000721, 0.0126545, 0.0004505, 0.0125083, -0.0125804, 0.0122039
7: -0.0192586, -0.0000012, -0.0197897, -0.0008421, -0.0145141, 0.0162647
8: 0.9686127, 1.0237877, 0.9670913, 1.0213785, -0.0527658, 0.0566964
9: -0.0093381, 0.0068779, -0.0086300, 0.0073251, -0.0155654, 0.0140663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344685, upper bound: 0.0338700
time: 0.98 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335095, upper bound: 0.0337021
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0002925, 0.0042740, -0.0040750, 0.0038566
1: -0.0000590, 0.0075128, -0.0000491, 0.0078717, -0.0079307, 0.0075618
2: 0.0050888, 0.0164284, 0.0045512, 0.0164135, -0.0113247, 0.0118771
3: -0.0068003, 0.0017265, -0.0072046, 0.0017153, -0.0085157, 0.0089311
4: -0.0106523, -0.0017216, -0.0110251, -0.0015766, -0.0090757, 0.0093035
5: 0.0011502, 0.0096616, 0.0007467, 0.0096504, -0.0085002, 0.0089148
6: 0.0004845, 0.0124988, -0.0000600, 0.0126511, -0.0121666, 0.0125588
7: -0.0193736, -0.0008966, -0.0193494, -0.0000207, -0.0157994, 0.0149349
8: 0.9682832, 1.0212221, 0.9683527, 1.0237318, -0.0554487, 0.0528694
9: -0.0085841, 0.0069748, -0.0093217, 0.0069543, -0.0143549, 0.0152474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338627, upper bound: 0.0344685
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0336872, upper bound: 0.0335303
time: 0.95 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0002925, 0.0042740, -0.0041172, 0.0038884
1: -0.0001261, 0.0075326, -0.0000491, 0.0078717, -0.0079978, 0.0075817
2: 0.0050591, 0.0165288, 0.0045512, 0.0164135, -0.0113543, 0.0119776
3: -0.0068226, 0.0018021, -0.0072046, 0.0017153, -0.0085380, 0.0090066
4: -0.0106728, -0.0017136, -0.0110251, -0.0015766, -0.0090962, 0.0093115
5: 0.0011280, 0.0097369, 0.0007467, 0.0096504, -0.0085224, 0.0089902
6: 0.0004544, 0.0125072, -0.0000600, 0.0126511, -0.0121967, 0.0125672
7: -0.0195373, -0.0008483, -0.0193494, -0.0000207, -0.0162315, 0.0152452
8: 0.9678143, 1.0213606, 0.9683527, 1.0237318, -0.0559175, 0.0530080
9: -0.0086248, 0.0071126, -0.0093217, 0.0069543, -0.0146161, 0.0155191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338627, upper bound: 0.0344816
time: 0.96 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0336872, upper bound: 0.0335303
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003058, 0.0042808, -0.0040957, 0.0038878
1: -0.0000590, 0.0075128, -0.0001113, 0.0078822, -0.0079412, 0.0076240
2: 0.0050888, 0.0164284, 0.0045355, 0.0165066, -0.0114178, 0.0118928
3: -0.0068003, 0.0017265, -0.0072164, 0.0017854, -0.0085857, 0.0089429
4: -0.0106523, -0.0017216, -0.0110360, -0.0015724, -0.0090799, 0.0093144
5: 0.0011502, 0.0096616, 0.0007349, 0.0097203, -0.0085700, 0.0089266
6: 0.0004845, 0.0124988, -0.0000758, 0.0126555, -0.0121711, 0.0125746
7: -0.0193736, -0.0008966, -0.0195011, 0.0000049, -0.0160452, 0.0152872
8: 0.9682832, 1.0212221, 0.9679180, 1.0238050, -0.0555218, 0.0533041
9: -0.0085841, 0.0069748, -0.0093432, 0.0070821, -0.0145632, 0.0154544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338611, upper bound: 0.0344685
time: 1.02 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0336853, upper bound: 0.0335095
time: 1.08 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003058, 0.0042808, -0.0040893, 0.0038764
1: -0.0001261, 0.0075326, -0.0001113, 0.0078822, -0.0080083, 0.0076438
2: 0.0050591, 0.0165288, 0.0045355, 0.0165066, -0.0114475, 0.0119933
3: -0.0068226, 0.0018021, -0.0072164, 0.0017854, -0.0086080, 0.0090184
4: -0.0106728, -0.0017136, -0.0110360, -0.0015724, -0.0091005, 0.0093224
5: 0.0011280, 0.0097369, 0.0007349, 0.0097203, -0.0085923, 0.0090020
6: 0.0004544, 0.0125072, -0.0000758, 0.0126555, -0.0122011, 0.0125830
7: -0.0195373, -0.0008483, -0.0195011, 0.0000049, -0.0158439, 0.0149978
8: 0.9678143, 1.0213606, 0.9679180, 1.0238050, -0.0559907, 0.0534427
9: -0.0086248, 0.0071126, -0.0093432, 0.0070821, -0.0144037, 0.0152611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0338611, upper bound: 0.0344816
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0336853, upper bound: 0.0335128
time: 0.91 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0002925, 0.0042740, -0.0039876, 0.0040215
1: 0.0000497, 0.0078693, -0.0000491, 0.0078717, -0.0078220, 0.0079184
2: 0.0045548, 0.0162656, 0.0045512, 0.0164135, -0.0118587, 0.0117143
3: -0.0072019, 0.0016041, -0.0072046, 0.0017153, -0.0089172, 0.0088087
4: -0.0110227, -0.0015776, -0.0110251, -0.0015766, -0.0094460, 0.0094476
5: 0.0007494, 0.0095394, 0.0007467, 0.0096504, -0.0089010, 0.0087926
6: -0.0000563, 0.0126500, -0.0000600, 0.0126511, -0.0127074, 0.0127100
7: -0.0191083, -0.0000265, -0.0193494, -0.0000207, -0.0147243, 0.0150851
8: 0.9690432, 1.0237151, 0.9683527, 1.0237318, -0.0546886, 0.0553625
9: -0.0093168, 0.0067514, -0.0093217, 0.0069543, -0.0146810, 0.0144643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344429, upper bound: 0.0336038
time: 0.91 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334979, upper bound: 0.0334730
time: 0.84 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0003058, 0.0042808, -0.0040100, 0.0040627
1: 0.0000497, 0.0078693, -0.0001113, 0.0078822, -0.0078325, 0.0079806
2: 0.0045548, 0.0162656, 0.0045355, 0.0165066, -0.0119518, 0.0117300
3: -0.0072019, 0.0016041, -0.0072164, 0.0017854, -0.0089872, 0.0088205
4: -0.0110227, -0.0015776, -0.0110360, -0.0015724, -0.0094503, 0.0094584
5: 0.0007494, 0.0095394, 0.0007349, 0.0097203, -0.0089709, 0.0088044
6: -0.0000563, 0.0126500, -0.0000758, 0.0126555, -0.0127118, 0.0127259
7: -0.0191083, -0.0000265, -0.0195011, 0.0000049, -0.0150719, 0.0154931
8: 0.9690432, 1.0237151, 0.9679180, 1.0238050, -0.0547618, 0.0557972
9: -0.0093168, 0.0067514, -0.0093432, 0.0070821, -0.0149120, 0.0147570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344429, upper bound: 0.0336038
time: 1.16 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334979, upper bound: 0.0334730
time: 1.26 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0002925, 0.0042740, -0.0040298, 0.0040406
1: -0.0000119, 0.0078797, -0.0000491, 0.0078717, -0.0078836, 0.0079288
2: 0.0045393, 0.0163578, 0.0045512, 0.0164135, -0.0118742, 0.0118066
3: -0.0072136, 0.0016735, -0.0072046, 0.0017153, -0.0089289, 0.0088780
4: -0.0110334, -0.0015734, -0.0110251, -0.0015766, -0.0094568, 0.0094518
5: 0.0007377, 0.0096086, 0.0007467, 0.0096504, -0.0089127, 0.0088619
6: -0.0000721, 0.0126545, -0.0000600, 0.0126511, -0.0127232, 0.0127144
7: -0.0192586, -0.0000012, -0.0193494, -0.0000207, -0.0151525, 0.0154051
8: 0.9686127, 1.0237877, 0.9683527, 1.0237318, -0.0551191, 0.0554351
9: -0.0093381, 0.0068779, -0.0093217, 0.0069543, -0.0149505, 0.0147244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344429, upper bound: 0.0336172
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334719, upper bound: 0.0334768
time: 1.00 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0003058, 0.0042808, -0.0040005, 0.0040321
1: -0.0000119, 0.0078797, -0.0001113, 0.0078822, -0.0078941, 0.0079910
2: 0.0045393, 0.0163578, 0.0045355, 0.0165066, -0.0119673, 0.0118222
3: -0.0072136, 0.0016735, -0.0072164, 0.0017854, -0.0089989, 0.0088898
4: -0.0110334, -0.0015734, -0.0110360, -0.0015724, -0.0094611, 0.0094626
5: 0.0007377, 0.0096086, 0.0007349, 0.0097203, -0.0089825, 0.0088736
6: -0.0000721, 0.0126545, -0.0000758, 0.0126555, -0.0127276, 0.0127303
7: -0.0192586, -0.0000012, -0.0195011, 0.0000049, -0.0147439, 0.0151201
8: 0.9686127, 1.0237877, 0.9679180, 1.0238050, -0.0551923, 0.0558698
9: -0.0093381, 0.0068779, -0.0093432, 0.0070821, -0.0147059, 0.0144532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344429, upper bound: 0.0336172
time: 1.10 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334719, upper bound: 0.0334768
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003178, 0.0041808, -0.0002946, 0.0040398, -0.0038494, 0.0039572
1: -0.0001673, 0.0077290, -0.0000590, 0.0075128, -0.0076801, 0.0077880
2: 0.0047651, 0.0165906, 0.0050888, 0.0164284, -0.0116633, 0.0115018
3: -0.0070438, 0.0018485, -0.0068003, 0.0017265, -0.0087703, 0.0086488
4: -0.0108768, -0.0016343, -0.0106523, -0.0017216, -0.0091552, 0.0090180
5: 0.0009072, 0.0097833, 0.0011502, 0.0096616, -0.0087543, 0.0086331
6: 0.0001566, 0.0125905, 0.0004845, 0.0124988, -0.0123422, 0.0121060
7: -0.0196380, -0.0003691, -0.0193736, -0.0008966, -0.0148764, 0.0150507
8: 0.9675258, 1.0227336, 0.9682832, 1.0212221, -0.0536963, 0.0544504
9: -0.0090283, 0.0071973, -0.0085841, 0.0069748, -0.0146170, 0.0143077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341876, upper bound: 0.0344756
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340695, upper bound: 0.0338547
time: 0.88 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003346, 0.0041981, -0.0002946, 0.0040398, -0.0038833, 0.0039935
1: -0.0002460, 0.0077554, -0.0000590, 0.0075128, -0.0077588, 0.0078144
2: 0.0047254, 0.0167084, 0.0050888, 0.0164284, -0.0117030, 0.0116196
3: -0.0070736, 0.0019371, -0.0068003, 0.0017265, -0.0088001, 0.0087374
4: -0.0109043, -0.0016236, -0.0106523, -0.0017216, -0.0091827, 0.0090287
5: 0.0008775, 0.0098718, 0.0011502, 0.0096616, -0.0087841, 0.0087215
6: 0.0001164, 0.0126017, 0.0004845, 0.0124988, -0.0123824, 0.0121173
7: -0.0198300, -0.0003045, -0.0193736, -0.0008966, -0.0152386, 0.0153304
8: 0.9669758, 1.0229187, 0.9682832, 1.0212221, -0.0542463, 0.0546355
9: -0.0090827, 0.0073590, -0.0085841, 0.0069748, -0.0148525, 0.0145401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341876, upper bound: 0.0344810
time: 0.91 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340695, upper bound: 0.0338547
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003178, 0.0041808, -0.0003089, 0.0040527, -0.0038811, 0.0039994
1: -0.0001673, 0.0077290, -0.0001261, 0.0075326, -0.0076999, 0.0078550
2: 0.0047651, 0.0165906, 0.0050591, 0.0165288, -0.0117637, 0.0115314
3: -0.0070438, 0.0018485, -0.0068226, 0.0018021, -0.0088459, 0.0086712
4: -0.0108768, -0.0016343, -0.0106728, -0.0017136, -0.0091632, 0.0090386
5: 0.0009072, 0.0097833, 0.0011280, 0.0097369, -0.0088297, 0.0086554
6: 0.0001566, 0.0125905, 0.0004544, 0.0125072, -0.0123506, 0.0121361
7: -0.0196380, -0.0003691, -0.0195373, -0.0008483, -0.0151866, 0.0154828
8: 0.9675258, 1.0227336, 0.9678143, 1.0213606, -0.0538348, 0.0549192
9: -0.0090283, 0.0071973, -0.0086248, 0.0071126, -0.0148887, 0.0145689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348433, upper bound: 0.0340055
time: 1.09 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340625, upper bound: 0.0338504
time: 1.33 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003346, 0.0041981, -0.0003089, 0.0040527, -0.0038699, 0.0039787
1: -0.0002460, 0.0077554, -0.0001261, 0.0075326, -0.0077786, 0.0078815
2: 0.0047254, 0.0167084, 0.0050591, 0.0165288, -0.0118034, 0.0116493
3: -0.0070736, 0.0019371, -0.0068226, 0.0018021, -0.0088757, 0.0087598
4: -0.0109043, -0.0016236, -0.0106728, -0.0017136, -0.0091907, 0.0090493
5: 0.0008775, 0.0098718, 0.0011280, 0.0097369, -0.0088595, 0.0087438
6: 0.0001164, 0.0126017, 0.0004544, 0.0125072, -0.0123908, 0.0121473
7: -0.0198300, -0.0003045, -0.0195373, -0.0008483, -0.0149411, 0.0150977
8: 0.9669758, 1.0229187, 0.9678143, 1.0213606, -0.0543848, 0.0551044
9: -0.0090827, 0.0073590, -0.0086248, 0.0071126, -0.0146327, 0.0143630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0341832, upper bound: 0.0344810
time: 1.01 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340625, upper bound: 0.0338507
time: 0.98 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003178, 0.0041808, -0.0002714, 0.0042724, -0.0041151, 0.0039659
1: -0.0001673, 0.0077290, 0.0000497, 0.0078693, -0.0080367, 0.0076792
2: 0.0047651, 0.0165906, 0.0045548, 0.0162656, -0.0115005, 0.0120358
3: -0.0070438, 0.0018485, -0.0072019, 0.0016041, -0.0086479, 0.0090504
4: -0.0108768, -0.0016343, -0.0110227, -0.0015776, -0.0092992, 0.0093884
5: 0.0009072, 0.0097833, 0.0007494, 0.0095394, -0.0086321, 0.0090339
6: 0.0001566, 0.0125905, -0.0000563, 0.0126500, -0.0124935, 0.0126468
7: -0.0196380, -0.0003691, -0.0191083, -0.0000265, -0.0162097, 0.0150043
8: 0.9675258, 1.0227336, 0.9690432, 1.0237151, -0.0561893, 0.0536904
9: -0.0090283, 0.0071973, -0.0093168, 0.0067514, -0.0144895, 0.0154305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338704, upper bound: 0.0340622
time: 1.04 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0336979, upper bound: 0.0332730
time: 1.17 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003346, 0.0041981, -0.0002714, 0.0042724, -0.0041490, 0.0040023
1: -0.0002460, 0.0077554, 0.0000497, 0.0078693, -0.0081153, 0.0077057
2: 0.0047254, 0.0167084, 0.0045548, 0.0162656, -0.0115402, 0.0121536
3: -0.0070736, 0.0019371, -0.0072019, 0.0016041, -0.0086777, 0.0091390
4: -0.0109043, -0.0016236, -0.0110227, -0.0015776, -0.0093267, 0.0093991
5: 0.0008775, 0.0098718, 0.0007494, 0.0095394, -0.0086619, 0.0091223
6: 0.0001164, 0.0126017, -0.0000563, 0.0126500, -0.0125336, 0.0126580
7: -0.0198300, -0.0003045, -0.0191083, -0.0000265, -0.0165720, 0.0152840
8: 0.9669758, 1.0229187, 0.9690432, 1.0237151, -0.0567393, 0.0538755
9: -0.0090827, 0.0073590, -0.0093168, 0.0067514, -0.0147250, 0.0156628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338704, upper bound: 0.0340650
time: 0.99 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0336979, upper bound: 0.0332730
time: 1.33 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003178, 0.0041808, -0.0002845, 0.0042792, -0.0041333, 0.0040019
1: -0.0001673, 0.0077290, -0.0000119, 0.0078797, -0.0080471, 0.0077408
2: 0.0047651, 0.0165906, 0.0045393, 0.0163578, -0.0115927, 0.0120513
3: -0.0070438, 0.0018485, -0.0072136, 0.0016735, -0.0087173, 0.0090621
4: -0.0108768, -0.0016343, -0.0110334, -0.0015734, -0.0093034, 0.0093992
5: 0.0009072, 0.0097833, 0.0007377, 0.0096086, -0.0087014, 0.0090456
6: 0.0001566, 0.0125905, -0.0000721, 0.0126545, -0.0124979, 0.0126626
7: -0.0196380, -0.0003691, -0.0192586, -0.0000012, -0.0163826, 0.0154117
8: 0.9675258, 1.0227336, 0.9686127, 1.0237877, -0.0562619, 0.0541208
9: -0.0090283, 0.0071973, -0.0093381, 0.0068779, -0.0147265, 0.0155761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338661, upper bound: 0.0340622
time: 1.04 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0336975, upper bound: 0.0332611
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003346, 0.0041981, -0.0002845, 0.0042792, -0.0041280, 0.0039873
1: -0.0002460, 0.0077554, -0.0000119, 0.0078797, -0.0081257, 0.0077673
2: 0.0047254, 0.0167084, 0.0045393, 0.0163578, -0.0116324, 0.0121692
3: -0.0070736, 0.0019371, -0.0072136, 0.0016735, -0.0087471, 0.0091507
4: -0.0109043, -0.0016236, -0.0110334, -0.0015734, -0.0093309, 0.0094099
5: 0.0008775, 0.0098718, 0.0007377, 0.0096086, -0.0087311, 0.0091340
6: 0.0001164, 0.0126017, -0.0000721, 0.0126545, -0.0125380, 0.0126738
7: -0.0198300, -0.0003045, -0.0192586, -0.0000012, -0.0162700, 0.0150489
8: 0.9669758, 1.0229187, 0.9686127, 1.0237877, -0.0568119, 0.0543060
9: -0.0090827, 0.0073590, -0.0093381, 0.0068779, -0.0145165, 0.0154820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338661, upper bound: 0.0340650
time: 1.01 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0336975, upper bound: 0.0332619
time: 1.00 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002896, 0.0043996, -0.0002946, 0.0040398, -0.0038520, 0.0042000
1: -0.0000357, 0.0080642, -0.0000590, 0.0075128, -0.0075485, 0.0081232
2: 0.0042630, 0.0163935, 0.0050888, 0.0164284, -0.0121654, 0.0113047
3: -0.0074213, 0.0017003, -0.0068003, 0.0017265, -0.0091478, 0.0085006
4: -0.0112251, -0.0014989, -0.0106523, -0.0017216, -0.0095035, 0.0091534
5: 0.0005304, 0.0096354, 0.0011502, 0.0096616, -0.0091312, 0.0084852
6: -0.0003518, 0.0127327, 0.0004845, 0.0124988, -0.0128506, 0.0122482
7: -0.0193168, 0.0004490, -0.0193736, -0.0008966, -0.0148542, 0.0162509
8: 0.9684460, 1.0250775, 0.9682832, 1.0212221, -0.0527761, 0.0567943
9: -0.0097171, 0.0069269, -0.0085841, 0.0069748, -0.0156275, 0.0142136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343946, upper bound: 0.0335868
time: 1.10 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334507, upper bound: 0.0334224
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002896, 0.0043996, -0.0003089, 0.0040527, -0.0038838, 0.0042422
1: -0.0000357, 0.0080642, -0.0001261, 0.0075326, -0.0075683, 0.0081903
2: 0.0042630, 0.0163935, 0.0050591, 0.0165288, -0.0122658, 0.0113343
3: -0.0074213, 0.0017003, -0.0068226, 0.0018021, -0.0092234, 0.0085229
4: -0.0112251, -0.0014989, -0.0106728, -0.0017136, -0.0095115, 0.0091740
5: 0.0005304, 0.0096354, 0.0011280, 0.0097369, -0.0092066, 0.0085074
6: -0.0003518, 0.0127327, 0.0004544, 0.0125072, -0.0128590, 0.0122783
7: -0.0193168, 0.0004490, -0.0195373, -0.0008483, -0.0151644, 0.0166830
8: 0.9684460, 1.0250775, 0.9678143, 1.0213606, -0.0529146, 0.0572631
9: -0.0097171, 0.0069269, -0.0086248, 0.0071126, -0.0158992, 0.0144748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343946, upper bound: 0.0335868
time: 1.42 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334507, upper bound: 0.0334224
time: 1.04 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003028, 0.0044103, -0.0002946, 0.0040398, -0.0038822, 0.0042199
1: -0.0000973, 0.0080806, -0.0000590, 0.0075128, -0.0076100, 0.0081396
2: 0.0042384, 0.0164857, 0.0050888, 0.0164284, -0.0121899, 0.0113968
3: -0.0074398, 0.0017696, -0.0068003, 0.0017265, -0.0091663, 0.0085699
4: -0.0112421, -0.0014922, -0.0106523, -0.0017216, -0.0095205, 0.0091600
5: 0.0005120, 0.0097046, 0.0011502, 0.0096616, -0.0091496, 0.0085543
6: -0.0003767, 0.0127397, 0.0004845, 0.0124988, -0.0128755, 0.0122552
7: -0.0194670, 0.0004890, -0.0193736, -0.0008966, -0.0152022, 0.0164500
8: 0.9680156, 1.0251920, 0.9682832, 1.0212221, -0.0532065, 0.0569088
9: -0.0097508, 0.0070534, -0.0085841, 0.0069748, -0.0157952, 0.0144365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343890, upper bound: 0.0336042
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334159, upper bound: 0.0334220
time: 0.94 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003028, 0.0044103, -0.0003089, 0.0040527, -0.0038719, 0.0042142
1: -0.0000973, 0.0080806, -0.0001261, 0.0075326, -0.0076299, 0.0082067
2: 0.0042384, 0.0164857, 0.0050591, 0.0165288, -0.0122904, 0.0114265
3: -0.0074398, 0.0017696, -0.0068226, 0.0018021, -0.0092418, 0.0085923
4: -0.0112421, -0.0014922, -0.0106728, -0.0017136, -0.0095285, 0.0091806
5: 0.0005120, 0.0097046, 0.0011280, 0.0097369, -0.0092250, 0.0085766
6: -0.0003767, 0.0127397, 0.0004544, 0.0125072, -0.0128839, 0.0122853
7: -0.0194670, 0.0004890, -0.0195373, -0.0008483, -0.0149182, 0.0162965
8: 0.9680156, 1.0251920, 0.9678143, 1.0213606, -0.0533450, 0.0573777
9: -0.0097508, 0.0070534, -0.0086248, 0.0071126, -0.0156422, 0.0142774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343890, upper bound: 0.0336042
time: 1.22 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334159, upper bound: 0.0334220
time: 1.56 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002896, 0.0043996, -0.0002714, 0.0042724, -0.0040270, 0.0041184
1: -0.0000357, 0.0080642, 0.0000497, 0.0078693, -0.0079051, 0.0080145
2: 0.0042630, 0.0163935, 0.0045548, 0.0162656, -0.0120026, 0.0118387
3: -0.0074213, 0.0017003, -0.0072019, 0.0016041, -0.0090254, 0.0089022
4: -0.0112251, -0.0014989, -0.0110227, -0.0015776, -0.0096475, 0.0095238
5: 0.0005304, 0.0096354, 0.0007494, 0.0095394, -0.0090090, 0.0088860
6: -0.0003518, 0.0127327, -0.0000563, 0.0126500, -0.0130019, 0.0127890
7: -0.0193168, 0.0004490, -0.0191083, -0.0000265, -0.0150772, 0.0152244
8: 0.9684460, 1.0250775, 0.9690432, 1.0237151, -0.0552691, 0.0560343
9: -0.0097171, 0.0069269, -0.0093168, 0.0067514, -0.0148854, 0.0145897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335146, upper bound: 0.0339551
time: 1.06 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333773, upper bound: 0.0331633
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003028, 0.0044103, -0.0002714, 0.0042724, -0.0040618, 0.0041432
1: -0.0000973, 0.0080806, 0.0000497, 0.0078693, -0.0079666, 0.0080309
2: 0.0042384, 0.0164857, 0.0045548, 0.0162656, -0.0120271, 0.0119308
3: -0.0074398, 0.0017696, -0.0072019, 0.0016041, -0.0090439, 0.0089715
4: -0.0112421, -0.0014922, -0.0110227, -0.0015776, -0.0096645, 0.0095304
5: 0.0005120, 0.0097046, 0.0007494, 0.0095394, -0.0090274, 0.0089552
6: -0.0003767, 0.0127397, -0.0000563, 0.0126500, -0.0130268, 0.0127960
7: -0.0194670, 0.0004890, -0.0191083, -0.0000265, -0.0154389, 0.0155147
8: 0.9680156, 1.0251920, 0.9690432, 1.0237151, -0.0556995, 0.0561488
9: -0.0097508, 0.0070534, -0.0093168, 0.0067514, -0.0151298, 0.0148128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335146, upper bound: 0.0339573
time: 0.99 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333773, upper bound: 0.0331633
time: 1.00 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002896, 0.0043996, -0.0002845, 0.0042792, -0.0040461, 0.0041606
1: -0.0000357, 0.0080642, -0.0000119, 0.0078797, -0.0079155, 0.0080761
2: 0.0042630, 0.0163935, 0.0045393, 0.0163578, -0.0120948, 0.0118542
3: -0.0074213, 0.0017003, -0.0072136, 0.0016735, -0.0090948, 0.0089139
4: -0.0112251, -0.0014989, -0.0110334, -0.0015734, -0.0096517, 0.0095346
5: 0.0005304, 0.0096354, 0.0007377, 0.0096086, -0.0090782, 0.0088976
6: -0.0003518, 0.0127327, -0.0000721, 0.0126545, -0.0130063, 0.0128048
7: -0.0193168, 0.0004490, -0.0192586, -0.0000012, -0.0153972, 0.0156526
8: 0.9684460, 1.0250775, 0.9686127, 1.0237877, -0.0553417, 0.0564647
9: -0.0097171, 0.0069269, -0.0093381, 0.0068779, -0.0151455, 0.0148592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A2_B2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335137, upper bound: 0.0339551
time: 0.97 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333712, upper bound: 0.0331541
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003028, 0.0044103, -0.0002845, 0.0042792, -0.0040378, 0.0041305
1: -0.0000973, 0.0080806, -0.0000119, 0.0078797, -0.0079770, 0.0080925
2: 0.0042384, 0.0164857, 0.0045393, 0.0163578, -0.0121193, 0.0119464
3: -0.0074398, 0.0017696, -0.0072136, 0.0016735, -0.0091132, 0.0089832
4: -0.0112421, -0.0014922, -0.0110334, -0.0015734, -0.0096687, 0.0095412
5: 0.0005120, 0.0097046, 0.0007377, 0.0096086, -0.0090966, 0.0089668
6: -0.0003767, 0.0127397, -0.0000721, 0.0126545, -0.0130312, 0.0128117
7: -0.0194670, 0.0004890, -0.0192586, -0.0000012, -0.0151100, 0.0152428
8: 0.9680156, 1.0251920, 0.9686127, 1.0237877, -0.0557721, 0.0565793
9: -0.0097508, 0.0070534, -0.0093381, 0.0068779, -0.0148734, 0.0146152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B1_A2_B2_A2_B2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0335137, upper bound: 0.0339573
time: 0.98 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333712, upper bound: 0.0331548
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003178, 0.0041808, -0.0039572, 0.0038494
1: -0.0000590, 0.0075128, -0.0001673, 0.0077290, -0.0077880, 0.0076801
2: 0.0050888, 0.0164284, 0.0047651, 0.0165906, -0.0115018, 0.0116633
3: -0.0068003, 0.0017265, -0.0070438, 0.0018485, -0.0086488, 0.0087703
4: -0.0106523, -0.0017216, -0.0108768, -0.0016343, -0.0090180, 0.0091552
5: 0.0011502, 0.0096616, 0.0009072, 0.0097833, -0.0086331, 0.0087543
6: 0.0004845, 0.0124988, 0.0001566, 0.0125905, -0.0121060, 0.0123422
7: -0.0193736, -0.0008966, -0.0196380, -0.0003691, -0.0150507, 0.0148764
8: 0.9682832, 1.0212221, 0.9675258, 1.0227336, -0.0544504, 0.0536963
9: -0.0085841, 0.0069748, -0.0090283, 0.0071973, -0.0143077, 0.0146170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344756, upper bound: 0.0341876
time: 1.04 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338547, upper bound: 0.0340695
time: 0.98 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003346, 0.0041981, -0.0039935, 0.0038833
1: -0.0000590, 0.0075128, -0.0002460, 0.0077554, -0.0078144, 0.0077588
2: 0.0050888, 0.0164284, 0.0047254, 0.0167084, -0.0116196, 0.0117030
3: -0.0068003, 0.0017265, -0.0070736, 0.0019371, -0.0087374, 0.0088001
4: -0.0106523, -0.0017216, -0.0109043, -0.0016236, -0.0090287, 0.0091827
5: 0.0011502, 0.0096616, 0.0008775, 0.0098718, -0.0087215, 0.0087841
6: 0.0004845, 0.0124988, 0.0001164, 0.0126017, -0.0121173, 0.0123824
7: -0.0193736, -0.0008966, -0.0198300, -0.0003045, -0.0153304, 0.0152386
8: 0.9682832, 1.0212221, 0.9669758, 1.0229187, -0.0546355, 0.0542463
9: -0.0085841, 0.0069748, -0.0090827, 0.0073590, -0.0145401, 0.0148525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344756, upper bound: 0.0341876
time: 0.87 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338547, upper bound: 0.0340695
time: 1.02 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003178, 0.0041808, -0.0039994, 0.0038811
1: -0.0001261, 0.0075326, -0.0001673, 0.0077290, -0.0078550, 0.0076999
2: 0.0050591, 0.0165288, 0.0047651, 0.0165906, -0.0115314, 0.0117637
3: -0.0068226, 0.0018021, -0.0070438, 0.0018485, -0.0086712, 0.0088459
4: -0.0106728, -0.0017136, -0.0108768, -0.0016343, -0.0090386, 0.0091632
5: 0.0011280, 0.0097369, 0.0009072, 0.0097833, -0.0086554, 0.0088297
6: 0.0004544, 0.0125072, 0.0001566, 0.0125905, -0.0121361, 0.0123506
7: -0.0195373, -0.0008483, -0.0196380, -0.0003691, -0.0154828, 0.0151866
8: 0.9678143, 1.0213606, 0.9675258, 1.0227336, -0.0549192, 0.0538348
9: -0.0086248, 0.0071126, -0.0090283, 0.0071973, -0.0145689, 0.0148887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0340055, upper bound: 0.0348621
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338504, upper bound: 0.0340640
time: 1.00 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003346, 0.0041981, -0.0039787, 0.0038699
1: -0.0001261, 0.0075326, -0.0002460, 0.0077554, -0.0078815, 0.0077786
2: 0.0050591, 0.0165288, 0.0047254, 0.0167084, -0.0116493, 0.0118034
3: -0.0068226, 0.0018021, -0.0070736, 0.0019371, -0.0087598, 0.0088757
4: -0.0106728, -0.0017136, -0.0109043, -0.0016236, -0.0090493, 0.0091907
5: 0.0011280, 0.0097369, 0.0008775, 0.0098718, -0.0087438, 0.0088595
6: 0.0004544, 0.0125072, 0.0001164, 0.0126017, -0.0121473, 0.0123908
7: -0.0195373, -0.0008483, -0.0198300, -0.0003045, -0.0150977, 0.0149411
8: 0.9678143, 1.0213606, 0.9669758, 1.0229187, -0.0551044, 0.0543848
9: -0.0086248, 0.0071126, -0.0090827, 0.0073590, -0.0143630, 0.0146327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0344756, upper bound: 0.0341885
time: 1.01 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0338504, upper bound: 0.0340640
time: 1.04 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0003178, 0.0041808, -0.0039659, 0.0041151
1: 0.0000497, 0.0078693, -0.0001673, 0.0077290, -0.0076792, 0.0080367
2: 0.0045548, 0.0162656, 0.0047651, 0.0165906, -0.0120358, 0.0115005
3: -0.0072019, 0.0016041, -0.0070438, 0.0018485, -0.0090504, 0.0086479
4: -0.0110227, -0.0015776, -0.0108768, -0.0016343, -0.0093884, 0.0092992
5: 0.0007494, 0.0095394, 0.0009072, 0.0097833, -0.0090339, 0.0086321
6: -0.0000563, 0.0126500, 0.0001566, 0.0125905, -0.0126468, 0.0124935
7: -0.0191083, -0.0000265, -0.0196380, -0.0003691, -0.0150043, 0.0162097
8: 0.9690432, 1.0237151, 0.9675258, 1.0227336, -0.0536904, 0.0561893
9: -0.0093168, 0.0067514, -0.0090283, 0.0071973, -0.0154305, 0.0144895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340622, upper bound: 0.0338704
time: 1.04 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332730, upper bound: 0.0336979
time: 1.06 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0003346, 0.0041981, -0.0040023, 0.0041490
1: 0.0000497, 0.0078693, -0.0002460, 0.0077554, -0.0077057, 0.0081153
2: 0.0045548, 0.0162656, 0.0047254, 0.0167084, -0.0121536, 0.0115402
3: -0.0072019, 0.0016041, -0.0070736, 0.0019371, -0.0091390, 0.0086777
4: -0.0110227, -0.0015776, -0.0109043, -0.0016236, -0.0093991, 0.0093267
5: 0.0007494, 0.0095394, 0.0008775, 0.0098718, -0.0091223, 0.0086619
6: -0.0000563, 0.0126500, 0.0001164, 0.0126017, -0.0126580, 0.0125336
7: -0.0191083, -0.0000265, -0.0198300, -0.0003045, -0.0152840, 0.0165720
8: 0.9690432, 1.0237151, 0.9669758, 1.0229187, -0.0538755, 0.0567393
9: -0.0093168, 0.0067514, -0.0090827, 0.0073590, -0.0156628, 0.0147250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340622, upper bound: 0.0338704
time: 1.12 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332730, upper bound: 0.0336979
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0003178, 0.0041808, -0.0040019, 0.0041333
1: -0.0000119, 0.0078797, -0.0001673, 0.0077290, -0.0077408, 0.0080471
2: 0.0045393, 0.0163578, 0.0047651, 0.0165906, -0.0120513, 0.0115927
3: -0.0072136, 0.0016735, -0.0070438, 0.0018485, -0.0090621, 0.0087173
4: -0.0110334, -0.0015734, -0.0108768, -0.0016343, -0.0093992, 0.0093034
5: 0.0007377, 0.0096086, 0.0009072, 0.0097833, -0.0090456, 0.0087014
6: -0.0000721, 0.0126545, 0.0001566, 0.0125905, -0.0126626, 0.0124979
7: -0.0192586, -0.0000012, -0.0196380, -0.0003691, -0.0154117, 0.0163826
8: 0.9686127, 1.0237877, 0.9675258, 1.0227336, -0.0541208, 0.0562619
9: -0.0093381, 0.0068779, -0.0090283, 0.0071973, -0.0155761, 0.0147265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340622, upper bound: 0.0338690
time: 1.01 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332611, upper bound: 0.0337050
time: 0.86 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0003346, 0.0041981, -0.0039873, 0.0041280
1: -0.0000119, 0.0078797, -0.0002460, 0.0077554, -0.0077673, 0.0081257
2: 0.0045393, 0.0163578, 0.0047254, 0.0167084, -0.0121692, 0.0116324
3: -0.0072136, 0.0016735, -0.0070736, 0.0019371, -0.0091507, 0.0087471
4: -0.0110334, -0.0015734, -0.0109043, -0.0016236, -0.0094099, 0.0093309
5: 0.0007377, 0.0096086, 0.0008775, 0.0098718, -0.0091340, 0.0087311
6: -0.0000721, 0.0126545, 0.0001164, 0.0126017, -0.0126738, 0.0125380
7: -0.0192586, -0.0000012, -0.0198300, -0.0003045, -0.0150489, 0.0162700
8: 0.9686127, 1.0237877, 0.9669758, 1.0229187, -0.0543060, 0.0568119
9: -0.0093381, 0.0068779, -0.0090827, 0.0073590, -0.0154820, 0.0145165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340622, upper bound: 0.0338690
time: 1.08 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332611, upper bound: 0.0337050
time: 1.03 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0002896, 0.0043996, -0.0042000, 0.0038520
1: -0.0000590, 0.0075128, -0.0000357, 0.0080642, -0.0081232, 0.0075485
2: 0.0050888, 0.0164284, 0.0042630, 0.0163935, -0.0113047, 0.0121654
3: -0.0068003, 0.0017265, -0.0074213, 0.0017003, -0.0085006, 0.0091478
4: -0.0106523, -0.0017216, -0.0112251, -0.0014989, -0.0091534, 0.0095035
5: 0.0011502, 0.0096616, 0.0005304, 0.0096354, -0.0084852, 0.0091312
6: 0.0004845, 0.0124988, -0.0003518, 0.0127327, -0.0122482, 0.0128506
7: -0.0193736, -0.0008966, -0.0193168, 0.0004490, -0.0162509, 0.0148542
8: 0.9682832, 1.0212221, 0.9684460, 1.0250775, -0.0567943, 0.0527761
9: -0.0085841, 0.0069748, -0.0097171, 0.0069269, -0.0142136, 0.0156275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0335868, upper bound: 0.0343946
time: 1.17 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334224, upper bound: 0.0334507
time: 1.37 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0002896, 0.0043996, -0.0042422, 0.0038838
1: -0.0001261, 0.0075326, -0.0000357, 0.0080642, -0.0081903, 0.0075683
2: 0.0050591, 0.0165288, 0.0042630, 0.0163935, -0.0113343, 0.0122658
3: -0.0068226, 0.0018021, -0.0074213, 0.0017003, -0.0085229, 0.0092234
4: -0.0106728, -0.0017136, -0.0112251, -0.0014989, -0.0091740, 0.0095115
5: 0.0011280, 0.0097369, 0.0005304, 0.0096354, -0.0085074, 0.0092066
6: 0.0004544, 0.0125072, -0.0003518, 0.0127327, -0.0122783, 0.0128590
7: -0.0195373, -0.0008483, -0.0193168, 0.0004490, -0.0166830, 0.0151644
8: 0.9678143, 1.0213606, 0.9684460, 1.0250775, -0.0572631, 0.0529146
9: -0.0086248, 0.0071126, -0.0097171, 0.0069269, -0.0144748, 0.0158992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0335868, upper bound: 0.0344059
time: 1.34 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334224, upper bound: 0.0334507
time: 1.35 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003028, 0.0044103, -0.0042199, 0.0038822
1: -0.0000590, 0.0075128, -0.0000973, 0.0080806, -0.0081396, 0.0076100
2: 0.0050888, 0.0164284, 0.0042384, 0.0164857, -0.0113968, 0.0121899
3: -0.0068003, 0.0017265, -0.0074398, 0.0017696, -0.0085699, 0.0091663
4: -0.0106523, -0.0017216, -0.0112421, -0.0014922, -0.0091600, 0.0095205
5: 0.0011502, 0.0096616, 0.0005120, 0.0097046, -0.0085543, 0.0091496
6: 0.0004845, 0.0124988, -0.0003767, 0.0127397, -0.0122552, 0.0128755
7: -0.0193736, -0.0008966, -0.0194670, 0.0004890, -0.0164500, 0.0152022
8: 0.9682832, 1.0212221, 0.9680156, 1.0251920, -0.0569088, 0.0532065
9: -0.0085841, 0.0069748, -0.0097508, 0.0070534, -0.0144365, 0.0157952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0335826, upper bound: 0.0343890
time: 0.98 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334086, upper bound: 0.0334159
time: 1.00 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003028, 0.0044103, -0.0042142, 0.0038719
1: -0.0001261, 0.0075326, -0.0000973, 0.0080806, -0.0082067, 0.0076299
2: 0.0050591, 0.0165288, 0.0042384, 0.0164857, -0.0114265, 0.0122904
3: -0.0068226, 0.0018021, -0.0074398, 0.0017696, -0.0085923, 0.0092418
4: -0.0106728, -0.0017136, -0.0112421, -0.0014922, -0.0091806, 0.0095285
5: 0.0011280, 0.0097369, 0.0005120, 0.0097046, -0.0085766, 0.0092250
6: 0.0004544, 0.0125072, -0.0003767, 0.0127397, -0.0122853, 0.0128839
7: -0.0195373, -0.0008483, -0.0194670, 0.0004890, -0.0162965, 0.0149182
8: 0.9678143, 1.0213606, 0.9680156, 1.0251920, -0.0573777, 0.0533450
9: -0.0086248, 0.0071126, -0.0097508, 0.0070534, -0.0142774, 0.0156422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0335826, upper bound: 0.0344059
time: 0.99 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0334086, upper bound: 0.0334211
time: 0.93 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0002896, 0.0043996, -0.0041184, 0.0040270
1: 0.0000497, 0.0078693, -0.0000357, 0.0080642, -0.0080145, 0.0079051
2: 0.0045548, 0.0162656, 0.0042630, 0.0163935, -0.0118387, 0.0120026
3: -0.0072019, 0.0016041, -0.0074213, 0.0017003, -0.0089022, 0.0090254
4: -0.0110227, -0.0015776, -0.0112251, -0.0014989, -0.0095238, 0.0096475
5: 0.0007494, 0.0095394, 0.0005304, 0.0096354, -0.0088860, 0.0090090
6: -0.0000563, 0.0126500, -0.0003518, 0.0127327, -0.0127890, 0.0130019
7: -0.0191083, -0.0000265, -0.0193168, 0.0004490, -0.0152244, 0.0150772
8: 0.9690432, 1.0237151, 0.9684460, 1.0250775, -0.0560343, 0.0552691
9: -0.0093168, 0.0067514, -0.0097171, 0.0069269, -0.0145897, 0.0148854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340216, upper bound: 0.0335146
time: 1.06 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0331647, upper bound: 0.0333773
time: 0.83 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002714, 0.0042724, -0.0003028, 0.0044103, -0.0041432, 0.0040618
1: 0.0000497, 0.0078693, -0.0000973, 0.0080806, -0.0080309, 0.0079666
2: 0.0045548, 0.0162656, 0.0042384, 0.0164857, -0.0119308, 0.0120271
3: -0.0072019, 0.0016041, -0.0074398, 0.0017696, -0.0089715, 0.0090439
4: -0.0110227, -0.0015776, -0.0112421, -0.0014922, -0.0095304, 0.0096645
5: 0.0007494, 0.0095394, 0.0005120, 0.0097046, -0.0089552, 0.0090274
6: -0.0000563, 0.0126500, -0.0003767, 0.0127397, -0.0127960, 0.0130268
7: -0.0191083, -0.0000265, -0.0194670, 0.0004890, -0.0155147, 0.0154389
8: 0.9690432, 1.0237151, 0.9680156, 1.0251920, -0.0561488, 0.0556995
9: -0.0093168, 0.0067514, -0.0097508, 0.0070534, -0.0148128, 0.0151298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340216, upper bound: 0.0335146
time: 1.09 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0331647, upper bound: 0.0333773
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0002896, 0.0043996, -0.0041606, 0.0040461
1: -0.0000119, 0.0078797, -0.0000357, 0.0080642, -0.0080761, 0.0079155
2: 0.0045393, 0.0163578, 0.0042630, 0.0163935, -0.0118542, 0.0120948
3: -0.0072136, 0.0016735, -0.0074213, 0.0017003, -0.0089139, 0.0090948
4: -0.0110334, -0.0015734, -0.0112251, -0.0014989, -0.0095346, 0.0096517
5: 0.0007377, 0.0096086, 0.0005304, 0.0096354, -0.0088976, 0.0090782
6: -0.0000721, 0.0126545, -0.0003518, 0.0127327, -0.0128048, 0.0130063
7: -0.0192586, -0.0000012, -0.0193168, 0.0004490, -0.0156526, 0.0153972
8: 0.9686127, 1.0237877, 0.9684460, 1.0250775, -0.0564647, 0.0553417
9: -0.0093381, 0.0068779, -0.0097171, 0.0069269, -0.0148592, 0.0151455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340216, upper bound: 0.0335262
time: 1.07 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0331576, upper bound: 0.0333809
time: 0.93 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002845, 0.0042792, -0.0003028, 0.0044103, -0.0041305, 0.0040378
1: -0.0000119, 0.0078797, -0.0000973, 0.0080806, -0.0080925, 0.0079770
2: 0.0045393, 0.0163578, 0.0042384, 0.0164857, -0.0119464, 0.0121193
3: -0.0072136, 0.0016735, -0.0074398, 0.0017696, -0.0089832, 0.0091132
4: -0.0110334, -0.0015734, -0.0112421, -0.0014922, -0.0095412, 0.0096687
5: 0.0007377, 0.0096086, 0.0005120, 0.0097046, -0.0089668, 0.0090966
6: -0.0000721, 0.0126545, -0.0003767, 0.0127397, -0.0128117, 0.0130312
7: -0.0192586, -0.0000012, -0.0194670, 0.0004890, -0.0152428, 0.0151100
8: 0.9686127, 1.0237877, 0.9680156, 1.0251920, -0.0565793, 0.0557721
9: -0.0093381, 0.0068779, -0.0097508, 0.0070534, -0.0146152, 0.0148734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340216, upper bound: 0.0335262
time: 1.13 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0331576, upper bound: 0.0333809
time: 0.97 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0007686, 0.0042353, -0.0005796, 0.0041177, -0.0042869, 0.0042401
1: -0.0004629, 0.0078124, -0.0003968, 0.0076321, -0.0080951, 0.0082092
2: 0.0046401, 0.0178297, 0.0049100, 0.0173714, -0.0127313, 0.0129197
3: -0.0071378, 0.0024037, -0.0069347, 0.0022288, -0.0093666, 0.0093384
4: -0.0109635, -0.0014617, -0.0107763, -0.0016418, -0.0093217, 0.0093146
5: 0.0008048, 0.0100742, 0.0010112, 0.0100184, -0.0092136, 0.0090630
6: 0.0000300, 0.0126259, 0.0003034, 0.0125494, -0.0125194, 0.0123225
7: -0.0205854, -0.0001654, -0.0203213, -0.0006053, -0.0154257, 0.0158170
8: 0.9655784, 1.0233171, 0.9659874, 1.0220567, -0.0564783, 0.0573297
9: -0.0092247, 0.0076359, -0.0088434, 0.0075755, -0.0152949, 0.0148644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367975, upper bound: 0.0364100
time: 1.06 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0367975, upper bound: 0.0364444
time: 1.20 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0007548, 0.0042302, -0.0019739, 0.0041329, -0.0042960, 0.0056027
1: -0.0004582, 0.0078047, -0.0008638, 0.0076555, -0.0081137, 0.0086685
2: 0.0046517, 0.0177965, 0.0048751, 0.0207585, -0.0153939, 0.0129214
3: -0.0071290, 0.0023910, -0.0069611, 0.0035267, -0.0106557, 0.0093521
4: -0.0109555, -0.0014812, -0.0108005, 0.0011510, -0.0121065, 0.0093193
5: 0.0008140, 0.0100702, 0.0009551, 0.0104198, -0.0096058, 0.0091151
6: 0.0000418, 0.0126226, 0.0002680, 0.0125593, -0.0125176, 0.0123546
7: -0.0205663, -0.0001843, -0.0222936, -0.0005483, -0.0154695, 0.0176215
8: 0.9656074, 1.0232630, 0.9630781, 1.0222199, -0.0566125, 0.0601849
9: -0.0092074, 0.0076317, -0.0089799, 0.0080311, -0.0156417, 0.0154221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0368238, upper bound: 0.0364100
time: 1.00 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0368238, upper bound: 0.0364444
time: 0.98 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0105105, 0.0041356, -0.0005796, 0.0041177, -0.0139993, 0.0041545
1: -0.0026277, 0.0076596, -0.0003968, 0.0076321, -0.0102598, 0.0080564
2: 0.0048689, 0.0413323, 0.0049100, 0.0173714, -0.0125025, 0.0353727
3: -0.0069657, 0.0112925, -0.0069347, 0.0022288, -0.0091945, 0.0172810
4: -0.0108048, 0.0199687, -0.0107763, -0.0016418, -0.0091630, 0.0307450
5: 0.0007352, 0.0119240, 0.0010112, 0.0100184, -0.0092832, 0.0104227
6: 0.0002617, 0.0125611, 0.0003034, 0.0125494, -0.0122877, 0.0122577
7: -0.0339228, -0.0005383, -0.0203213, -0.0006053, -0.0282305, 0.0156007
8: 0.9512550, 1.0222487, 0.9659874, 1.0220567, -0.0708017, 0.0562614
9: -0.0097101, 0.0097036, -0.0088434, 0.0075755, -0.0172856, 0.0157200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B2_A2_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0337520, upper bound: 0.0340348
time: 1.05 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333654, upper bound: 0.0339868
time: 1.71 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0104949, 0.0041313, -0.0019739, 0.0041329, -0.0140056, 0.0055173
1: -0.0026243, 0.0076531, -0.0008638, 0.0076555, -0.0102797, 0.0085169
2: 0.0048787, 0.0412945, 0.0048751, 0.0207585, -0.0152568, 0.0353657
3: -0.0069584, 0.0112781, -0.0069611, 0.0035267, -0.0104850, 0.0172902
4: -0.0107980, 0.0199342, -0.0108005, 0.0011510, -0.0119491, 0.0307347
5: 0.0007436, 0.0119211, 0.0009551, 0.0104198, -0.0096762, 0.0106087
6: 0.0002716, 0.0125583, 0.0002680, 0.0125593, -0.0122877, 0.0122903
7: -0.0339011, -0.0005542, -0.0222936, -0.0005483, -0.0282641, 0.0173982
8: 0.9512785, 1.0222032, 0.9630781, 1.0222199, -0.0709414, 0.0591251
9: -0.0096956, 0.0097003, -0.0089799, 0.0080311, -0.0177267, 0.0162798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_B2_A2_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0337673, upper bound: 0.0340348
time: 1.17 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333652, upper bound: 0.0339868
time: 1.00 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0005796, 0.0041177, -0.0105105, 0.0041356, -0.0041545, 0.0139993
1: -0.0003968, 0.0076321, -0.0026277, 0.0076596, -0.0080564, 0.0102598
2: 0.0049100, 0.0173714, 0.0048689, 0.0413323, -0.0353727, 0.0125025
3: -0.0069347, 0.0022288, -0.0069657, 0.0112925, -0.0172810, 0.0091945
4: -0.0107763, -0.0016418, -0.0108048, 0.0199687, -0.0307450, 0.0091630
5: 0.0010112, 0.0100184, 0.0007352, 0.0119240, -0.0104227, 0.0092832
6: 0.0003034, 0.0125494, 0.0002617, 0.0125611, -0.0122577, 0.0122877
7: -0.0203213, -0.0006053, -0.0339228, -0.0005383, -0.0156007, 0.0282305
8: 0.9659874, 1.0220567, 0.9512550, 1.0222487, -0.0562614, 0.0708017
9: -0.0088434, 0.0075755, -0.0097101, 0.0097036, -0.0157200, 0.0172856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332849, upper bound: 0.0333938
time: 0.99 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0331853, upper bound: 0.0331296
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0103240, 0.0040144, -0.0105105, 0.0041356, -0.0138285, 0.0138871
1: -0.0025680, 0.0074739, -0.0026277, 0.0076596, -0.0102276, 0.0101016
2: 0.0051471, 0.0408788, 0.0048689, 0.0413323, -0.0350011, 0.0348166
3: -0.0067565, 0.0111193, -0.0069657, 0.0112925, -0.0169880, 0.0170227
4: -0.0106119, 0.0195539, -0.0108048, 0.0199687, -0.0305806, 0.0303587
5: 0.0009396, 0.0118735, 0.0007352, 0.0119240, -0.0109845, 0.0111383
6: 0.0005434, 0.0124823, 0.0002617, 0.0125611, -0.0120176, 0.0122206
7: -0.0336609, -0.0009915, -0.0339228, -0.0005383, -0.0278665, 0.0275995
8: 0.9515460, 1.0209502, 0.9512550, 1.0222487, -0.0707028, 0.0696952
9: -0.0093437, 0.0096484, -0.0097101, 0.0097036, -0.0190473, 0.0193584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
time: 1.34 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
time: 1.23 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0019739, 0.0041329, -0.0104949, 0.0041313, -0.0055173, 0.0140056
1: -0.0008638, 0.0076555, -0.0026243, 0.0076531, -0.0085169, 0.0102797
2: 0.0048751, 0.0207585, 0.0048787, 0.0412945, -0.0353657, 0.0152568
3: -0.0069611, 0.0035267, -0.0069584, 0.0112781, -0.0172902, 0.0104850
4: -0.0108005, 0.0011510, -0.0107980, 0.0199342, -0.0307347, 0.0119491
5: 0.0009551, 0.0104198, 0.0007436, 0.0119211, -0.0106087, 0.0096762
6: 0.0002680, 0.0125593, 0.0002716, 0.0125583, -0.0122903, 0.0122877
7: -0.0222936, -0.0005483, -0.0339011, -0.0005542, -0.0173982, 0.0282641
8: 0.9630781, 1.0222199, 0.9512785, 1.0222032, -0.0591251, 0.0709414
9: -0.0089799, 0.0080311, -0.0096956, 0.0097003, -0.0162798, 0.0177267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 78

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332708, upper bound: 0.0334032
time: 1.11 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0331798, upper bound: 0.0331226
time: 1.27 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0117704, 0.0040339, -0.0104949, 0.0041313, -0.0152803, 0.0138945
1: -0.0028126, 0.0075037, -0.0026243, 0.0076531, -0.0104657, 0.0101280
2: 0.0051024, 0.0444233, 0.0048787, 0.0412945, -0.0349928, 0.0386445
3: -0.0067901, 0.0124899, -0.0069584, 0.0112781, -0.0169963, 0.0186038
4: -0.0106428, 0.0227793, -0.0107980, 0.0199342, -0.0305770, 0.0335773
5: 0.0008732, 0.0120954, 0.0007436, 0.0119211, -0.0110479, 0.0113519
6: 0.0004982, 0.0124949, 0.0002716, 0.0125583, -0.0120601, 0.0122233
7: -0.0357505, -0.0009187, -0.0339011, -0.0005542, -0.0302502, 0.0276312
8: 0.9491154, 1.0211587, 0.9512785, 1.0222032, -0.0730878, 0.0698802
9: -0.0096065, 0.0099348, -0.0096956, 0.0097003, -0.0193068, 0.0196303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598
time: 1.05 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598
time: 1.01 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.82 seconds
IS_B1_A1_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0347265, upper bound: 0.0341323
IS_B1_A1_A2_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340202, upper bound: 0.0340208
IS_B1_A1_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0347265, upper bound: 0.0341323
IS_B1_A1_A2_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340202, upper bound: 0.0340208
IS_B1_A1_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0347262, upper bound: 0.0341326
IS_B1_A1_A2_B1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340157, upper bound: 0.0340200
IS_B1_A1_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0347262, upper bound: 0.0341326
IS_B1_A1_A2_B1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340157, upper bound: 0.0340200
IS_B1_A1_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344685, upper bound: 0.0338627
IS_B1_A1_A2_B1_A2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335303, upper bound: 0.0336872
IS_B1_A1_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344685, upper bound: 0.0338628
IS_B1_A1_A2_B1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335303, upper bound: 0.0336872
IS_B1_A1_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344685, upper bound: 0.0338700
IS_B1_A1_A2_B1_A2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335095, upper bound: 0.0337021
IS_B1_A1_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344685, upper bound: 0.0338700
IS_B1_A1_A2_B1_A2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335095, upper bound: 0.0337021
IS_B1_A1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338627, upper bound: 0.0344685
IS_B1_A1_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0336872, upper bound: 0.0335303
IS_B1_A1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338627, upper bound: 0.0344816
IS_B1_A1_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0336872, upper bound: 0.0335303
IS_B1_A1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338611, upper bound: 0.0344685
IS_B1_A1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0336853, upper bound: 0.0335095
IS_B1_A1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338611, upper bound: 0.0344816
IS_B1_A1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0336853, upper bound: 0.0335128
IS_B1_A1_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344429, upper bound: 0.0336038
IS_B1_A1_A2_B2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334979, upper bound: 0.0334730
IS_B1_A1_A2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344429, upper bound: 0.0336038
IS_B1_A1_A2_B2_A2_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334979, upper bound: 0.0334730
IS_B1_A1_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344429, upper bound: 0.0336172
IS_B1_A1_A2_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334719, upper bound: 0.0334768
IS_B1_A1_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344429, upper bound: 0.0336172
IS_B1_A1_A2_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334719, upper bound: 0.0334768
IS_B1_A2_B2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0341876, upper bound: 0.0344756
IS_B1_A2_B2_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340695, upper bound: 0.0338547
IS_B1_A2_B2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0341876, upper bound: 0.0344810
IS_B1_A2_B2_A1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340695, upper bound: 0.0338547
IS_B1_A2_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0348433, upper bound: 0.0340055
IS_B1_A2_B2_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340625, upper bound: 0.0338504
IS_B1_A2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0341832, upper bound: 0.0344810
IS_B1_A2_B2_A1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340625, upper bound: 0.0338507
IS_B1_A2_B2_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338704, upper bound: 0.0340622
IS_B1_A2_B2_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0336979, upper bound: 0.0332730
IS_B1_A2_B2_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338704, upper bound: 0.0340650
IS_B1_A2_B2_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0336979, upper bound: 0.0332730
IS_B1_A2_B2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338661, upper bound: 0.0340622
IS_B1_A2_B2_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0336975, upper bound: 0.0332611
IS_B1_A2_B2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338661, upper bound: 0.0340650
IS_B1_A2_B2_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0336975, upper bound: 0.0332619
IS_B1_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0343946, upper bound: 0.0335868
IS_B1_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334507, upper bound: 0.0334224
IS_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0343946, upper bound: 0.0335868
IS_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334507, upper bound: 0.0334224
IS_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0343890, upper bound: 0.0336042
IS_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334159, upper bound: 0.0334220
IS_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0343890, upper bound: 0.0336042
IS_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334159, upper bound: 0.0334220
IS_B1_A2_B2_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335146, upper bound: 0.0339551
IS_B1_A2_B2_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0333773, upper bound: 0.0331633
IS_B1_A2_B2_A2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335146, upper bound: 0.0339573
IS_B1_A2_B2_A2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0333773, upper bound: 0.0331633
IS_B1_A2_B2_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335137, upper bound: 0.0339551
IS_B1_A2_B2_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0333712, upper bound: 0.0331541
IS_B1_A2_B2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335137, upper bound: 0.0339573
IS_B1_A2_B2_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0333712, upper bound: 0.0331548
IS_B2_A2_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344756, upper bound: 0.0341876
IS_B2_A2_A1_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338547, upper bound: 0.0340695
IS_B2_A2_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344756, upper bound: 0.0341876
IS_B2_A2_A1_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338547, upper bound: 0.0340695
IS_B2_A2_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340055, upper bound: 0.0348621
IS_B2_A2_A1_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338504, upper bound: 0.0340640
IS_B2_A2_A1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0344756, upper bound: 0.0341885
IS_B2_A2_A1_B1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0338504, upper bound: 0.0340640
IS_B2_A2_A1_B1_A2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340622, upper bound: 0.0338704
IS_B2_A2_A1_B1_A2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0332730, upper bound: 0.0336979
IS_B2_A2_A1_B1_A2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340622, upper bound: 0.0338704
IS_B2_A2_A1_B1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0332730, upper bound: 0.0336979
IS_B2_A2_A1_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340622, upper bound: 0.0338690
IS_B2_A2_A1_B1_A2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0332611, upper bound: 0.0337050
IS_B2_A2_A1_B1_A2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340622, upper bound: 0.0338690
IS_B2_A2_A1_B1_A2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0332611, upper bound: 0.0337050
IS_B2_A2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335868, upper bound: 0.0343946
IS_B2_A2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334224, upper bound: 0.0334507
IS_B2_A2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335868, upper bound: 0.0344059
IS_B2_A2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334224, upper bound: 0.0334507
IS_B2_A2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335826, upper bound: 0.0343890
IS_B2_A2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334086, upper bound: 0.0334159
IS_B2_A2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0335826, upper bound: 0.0344059
IS_B2_A2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0334086, upper bound: 0.0334211
IS_B2_A2_A1_B2_A2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340216, upper bound: 0.0335146
IS_B2_A2_A1_B2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0331647, upper bound: 0.0333773
IS_B2_A2_A1_B2_A2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340216, upper bound: 0.0335146
IS_B2_A2_A1_B2_A2_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0331647, upper bound: 0.0333773
IS_B2_A2_A1_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340216, upper bound: 0.0335262
IS_B2_A2_A1_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0331576, upper bound: 0.0333809
IS_B2_A2_A1_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0340216, upper bound: 0.0335262
IS_B2_A2_A1_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0331576, upper bound: 0.0333809
IS_B2_A2_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0367975, upper bound: 0.0364100
IS_B2_A2_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0367975, upper bound: 0.0364444
IS_B2_A2_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0368238, upper bound: 0.0364100
IS_B2_A2_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0368238, upper bound: 0.0364444
IS_B2_A2_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0337520, upper bound: 0.0340348
IS_B2_A2_A2_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0333654, upper bound: 0.0339868
IS_B2_A2_A2_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0337673, upper bound: 0.0340348
IS_B2_A2_A2_B2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0333652, upper bound: 0.0339868
IS_B2_A2_A2_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0332849, upper bound: 0.0333938
IS_B2_A2_A2_B2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0331853, upper bound: 0.0331296
IS_B2_A2_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
IS_B2_A2_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0351141, upper bound: 0.0349754
IS_B2_A2_A2_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0332708, upper bound: 0.0334032
IS_B2_A2_A2_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0331798, upper bound: 0.0331226
IS_B2_A2_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598
IS_B2_A2_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.82
Output dim: 8, lower bound: -0.0351089, upper bound: 0.0349598

## BFS IS instance: IS_B1_A1_A2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002938, 0.0039692, -0.0003165, 0.0040414, -0.0038100, 0.0037701
1: -0.0000552, 0.0074046, -0.0001614, 0.0075153, -0.0075704, 0.0075660
2: 0.0052509, 0.0164226, 0.0050850, 0.0165817, -0.0113309, 0.0113376
3: -0.0066785, 0.0017222, -0.0068032, 0.0018419, -0.0085203, 0.0085254
4: -0.0105399, -0.0017653, -0.0106549, -0.0017206, -0.0088193, 0.0088896
5: 0.0012719, 0.0096572, 0.0011474, 0.0097767, -0.0085048, 0.0085098
6: 0.0006486, 0.0124529, 0.0004806, 0.0124999, -0.0118513, 0.0119723
7: -0.0193642, -0.0011607, -0.0196236, -0.0008905, -0.0145000, 0.0145977
8: 0.9683101, 1.0204656, 0.9675671, 1.0212396, -0.0529295, 0.0528985
9: -0.0083617, 0.0069668, -0.0085892, 0.0071852, -0.0141646, 0.0141103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333796, upper bound: 0.0309229
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333796, upper bound: 0.0309229
time: 0.86 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002938, 0.0039692, -0.0003310, 0.0040544, -0.0038440, 0.0038114
1: -0.0000552, 0.0074046, -0.0002295, 0.0075351, -0.0075903, 0.0076341
2: 0.0052509, 0.0164226, 0.0050553, 0.0166837, -0.0114328, 0.0113673
3: -0.0066785, 0.0017222, -0.0068255, 0.0019185, -0.0085970, 0.0085477
4: -0.0105399, -0.0017653, -0.0106755, -0.0017126, -0.0088273, 0.0089102
5: 0.0012719, 0.0096572, 0.0011251, 0.0098532, -0.0085813, 0.0085321
6: 0.0006486, 0.0124529, 0.0004505, 0.0125083, -0.0118597, 0.0120024
7: -0.0193642, -0.0011607, -0.0197897, -0.0008421, -0.0148557, 0.0150047
8: 0.9683101, 1.0204656, 0.9670913, 1.0213785, -0.0530684, 0.0533743
9: -0.0083617, 0.0069668, -0.0086300, 0.0073251, -0.0144006, 0.0144099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332749, upper bound: 0.0306599
time: 1.06 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332749, upper bound: 0.0341322
time: 1.09 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003081, 0.0039823, -0.0003165, 0.0040414, -0.0038522, 0.0038011
1: -0.0001223, 0.0074247, -0.0001614, 0.0075153, -0.0076376, 0.0075862
2: 0.0052206, 0.0165231, 0.0050850, 0.0165817, -0.0113611, 0.0114381
3: -0.0067012, 0.0017978, -0.0068032, 0.0018419, -0.0085431, 0.0086010
4: -0.0105608, -0.0017571, -0.0106549, -0.0017206, -0.0088402, 0.0088977
5: 0.0012492, 0.0097327, 0.0011474, 0.0097767, -0.0085275, 0.0085853
6: 0.0006180, 0.0124615, 0.0004806, 0.0124999, -0.0118819, 0.0119808
7: -0.0195280, -0.0011114, -0.0196236, -0.0008905, -0.0149326, 0.0148946
8: 0.9678407, 1.0206065, 0.9675671, 1.0212396, -0.0533990, 0.0530394
9: -0.0084032, 0.0071048, -0.0085892, 0.0071852, -0.0144146, 0.0143837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333246, upper bound: 0.0308986
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0333246, upper bound: 0.0341345
time: 0.87 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003081, 0.0039823, -0.0003310, 0.0040544, -0.0038300, 0.0037898
1: -0.0001223, 0.0074247, -0.0002295, 0.0075351, -0.0076574, 0.0076542
2: 0.0052206, 0.0165231, 0.0050553, 0.0166837, -0.0114630, 0.0114678
3: -0.0067012, 0.0017978, -0.0068255, 0.0019185, -0.0086197, 0.0086233
4: -0.0105608, -0.0017571, -0.0106755, -0.0017126, -0.0088483, 0.0089183
5: 0.0012492, 0.0097327, 0.0011251, 0.0098532, -0.0086040, 0.0086076
6: 0.0006180, 0.0124615, 0.0004505, 0.0125083, -0.0118903, 0.0120109
7: -0.0195280, -0.0011114, -0.0197897, -0.0008421, -0.0145497, 0.0146633
8: 0.9678407, 1.0206065, 0.9670913, 1.0213785, -0.0535378, 0.0535152
9: -0.0084032, 0.0071048, -0.0086300, 0.0073251, -0.0142169, 0.0141290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332677, upper bound: 0.0306572
time: 0.99 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0332677, upper bound: 0.0341326
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002706, 0.0042002, -0.0003165, 0.0040414, -0.0038188, 0.0040335
1: 0.0000533, 0.0077585, -0.0001614, 0.0075153, -0.0074620, 0.0079200
2: 0.0047207, 0.0162602, 0.0050850, 0.0165817, -0.0118610, 0.0111752
3: -0.0070771, 0.0016001, -0.0068032, 0.0018419, -0.0089190, 0.0084033
4: -0.0109076, -0.0016223, -0.0106549, -0.0017206, -0.0091870, 0.0090326
5: 0.0008740, 0.0095354, 0.0011474, 0.0097767, -0.0089027, 0.0083880
6: 0.0001117, 0.0126031, 0.0004806, 0.0124999, -0.0123882, 0.0121224
7: -0.0190997, -0.0002969, -0.0196236, -0.0008905, -0.0144543, 0.0159269
8: 0.9690681, 1.0229405, 0.9675671, 1.0212396, -0.0521715, 0.0553734
9: -0.0090891, 0.0067441, -0.0085892, 0.0071852, -0.0152839, 0.0139826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0328644, upper bound: 0.0306440
time: 0.90 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0328644, upper bound: 0.0338803
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002706, 0.0042002, -0.0003310, 0.0040544, -0.0038528, 0.0040748
1: 0.0000533, 0.0077585, -0.0002295, 0.0075351, -0.0074819, 0.0079880
2: 0.0047207, 0.0162602, 0.0050553, 0.0166837, -0.0119629, 0.0112049
3: -0.0070771, 0.0016001, -0.0068255, 0.0019185, -0.0089956, 0.0084256
4: -0.0109076, -0.0016223, -0.0106755, -0.0017126, -0.0091950, 0.0090532
5: 0.0008740, 0.0095354, 0.0011251, 0.0098532, -0.0089792, 0.0084103
6: 0.0001117, 0.0126031, 0.0004505, 0.0125083, -0.0123966, 0.0121525
7: -0.0190997, -0.0002969, -0.0197897, -0.0008421, -0.0148101, 0.0163340
8: 0.9690681, 1.0229405, 0.9670913, 1.0213785, -0.0523104, 0.0558492
9: -0.0090891, 0.0067441, -0.0086300, 0.0073251, -0.0155199, 0.0142822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0327310, upper bound: 0.0303682
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0327310, upper bound: 0.0338574
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002838, 0.0042071, -0.0003165, 0.0040414, -0.0038548, 0.0040521
1: -0.0000084, 0.0077691, -0.0001614, 0.0075153, -0.0075237, 0.0079306
2: 0.0047049, 0.0163526, 0.0050850, 0.0165817, -0.0118769, 0.0112676
3: -0.0070890, 0.0016696, -0.0068032, 0.0018419, -0.0089309, 0.0084727
4: -0.0109186, -0.0016180, -0.0106549, -0.0017206, -0.0091980, 0.0090368
5: 0.0008620, 0.0096047, 0.0011474, 0.0097767, -0.0089146, 0.0084573
6: 0.0000956, 0.0126076, 0.0004806, 0.0124999, -0.0124042, 0.0121269
7: -0.0192502, -0.0002710, -0.0196236, -0.0008905, -0.0148625, 0.0161081
8: 0.9686368, 1.0230147, 0.9675671, 1.0212396, -0.0526028, 0.0554475
9: -0.0091109, 0.0068708, -0.0085892, 0.0071852, -0.0154364, 0.0142208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0328029, upper bound: 0.0305658
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0328029, upper bound: 0.0338699
time: 1.04 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002838, 0.0042071, -0.0003310, 0.0040544, -0.0038385, 0.0040467
1: -0.0000084, 0.0077691, -0.0002295, 0.0075351, -0.0075436, 0.0079986
2: 0.0047049, 0.0163526, 0.0050553, 0.0166837, -0.0119788, 0.0112973
3: -0.0070890, 0.0016696, -0.0068255, 0.0019185, -0.0090076, 0.0084951
4: -0.0109186, -0.0016180, -0.0106755, -0.0017126, -0.0092060, 0.0090574
5: 0.0008620, 0.0096047, 0.0011251, 0.0098532, -0.0089912, 0.0084796
6: 0.0000956, 0.0126076, 0.0004505, 0.0125083, -0.0124127, 0.0121570
7: -0.0192502, -0.0002710, -0.0197897, -0.0008421, -0.0145017, 0.0159872
8: 0.9686368, 1.0230147, 0.9670913, 1.0213785, -0.0527417, 0.0559233
9: -0.0091109, 0.0068708, -0.0086300, 0.0073251, -0.0153317, 0.0140131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0327306, upper bound: 0.0303608
time: 1.08 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0327306, upper bound: 0.0338687
time: 1.46 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0002917, 0.0042016, -0.0040022, 0.0038560
1: -0.0000590, 0.0075128, -0.0000455, 0.0077609, -0.0078199, 0.0075583
2: 0.0050888, 0.0164284, 0.0047173, 0.0164081, -0.0113193, 0.0117111
3: -0.0068003, 0.0017265, -0.0070797, 0.0017113, -0.0085116, 0.0088063
4: -0.0106523, -0.0017216, -0.0109100, -0.0016214, -0.0090309, 0.0091884
5: 0.0011502, 0.0096616, 0.0008714, 0.0096464, -0.0084961, 0.0087902
6: 0.0004845, 0.0124988, 0.0001082, 0.0126040, -0.0121196, 0.0123906
7: -0.0193736, -0.0008966, -0.0193406, -0.0002912, -0.0155171, 0.0149225
8: 0.9682832, 1.0212221, 0.9683776, 1.0229566, -0.0546734, 0.0528445
9: -0.0085841, 0.0069748, -0.0090939, 0.0069470, -0.0143027, 0.0150097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0321519, upper bound: 0.0313857
time: 0.78 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0321519, upper bound: 0.0344684
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0002917, 0.0042016, -0.0040444, 0.0038878
1: -0.0001261, 0.0075326, -0.0000455, 0.0077609, -0.0078869, 0.0075781
2: 0.0050591, 0.0165288, 0.0047173, 0.0164081, -0.0113490, 0.0118115
3: -0.0068226, 0.0018021, -0.0070797, 0.0017113, -0.0085339, 0.0088818
4: -0.0106728, -0.0017136, -0.0109100, -0.0016214, -0.0090515, 0.0091964
5: 0.0011280, 0.0097369, 0.0008714, 0.0096464, -0.0085184, 0.0088656
6: 0.0004544, 0.0125072, 0.0001082, 0.0126040, -0.0121496, 0.0123990
7: -0.0195373, -0.0008483, -0.0193406, -0.0002912, -0.0159492, 0.0152327
8: 0.9678143, 1.0213606, 0.9683776, 1.0229566, -0.0551423, 0.0529830
9: -0.0086248, 0.0071126, -0.0090939, 0.0069470, -0.0145639, 0.0152814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0320508, upper bound: 0.0313017
time: 1.00 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320508, upper bound: 0.0344579
time: 1.04 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003050, 0.0042086, -0.0040241, 0.0038872
1: -0.0000590, 0.0075128, -0.0001078, 0.0077716, -0.0078306, 0.0076206
2: 0.0050888, 0.0164284, 0.0047012, 0.0165014, -0.0114126, 0.0117271
3: -0.0068003, 0.0017265, -0.0070918, 0.0017815, -0.0085818, 0.0088183
4: -0.0106523, -0.0017216, -0.0109211, -0.0016171, -0.0090352, 0.0091995
5: 0.0011502, 0.0096616, 0.0008593, 0.0097164, -0.0085662, 0.0088022
6: 0.0004845, 0.0124988, 0.0000919, 0.0126086, -0.0121241, 0.0124069
7: -0.0193736, -0.0008966, -0.0194927, -0.0002651, -0.0157781, 0.0152754
8: 0.9682832, 1.0212221, 0.9679420, 1.0230315, -0.0547483, 0.0532801
9: -0.0085841, 0.0069748, -0.0091159, 0.0070750, -0.0145100, 0.0152295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0320476, upper bound: 0.0312512
time: 0.78 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320476, upper bound: 0.0344467
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003050, 0.0042086, -0.0040176, 0.0038759
1: -0.0001261, 0.0075326, -0.0001078, 0.0077716, -0.0078976, 0.0076404
2: 0.0050591, 0.0165288, 0.0047012, 0.0165014, -0.0114423, 0.0118276
3: -0.0068226, 0.0018021, -0.0070918, 0.0017815, -0.0086041, 0.0088938
4: -0.0106728, -0.0017136, -0.0109211, -0.0016171, -0.0090558, 0.0092075
5: 0.0011280, 0.0097369, 0.0008593, 0.0097164, -0.0085884, 0.0088776
6: 0.0004544, 0.0125072, 0.0000919, 0.0126086, -0.0121542, 0.0124153
7: -0.0195373, -0.0008483, -0.0194927, -0.0002651, -0.0155640, 0.0149853
8: 0.9678143, 1.0213606, 0.9679420, 1.0230315, -0.0552171, 0.0534186
9: -0.0086248, 0.0071126, -0.0091159, 0.0070750, -0.0143521, 0.0150254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0320076, upper bound: 0.0312332
time: 1.27 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320076, upper bound: 0.0344579
time: 1.05 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002706, 0.0042002, -0.0002925, 0.0042740, -0.0039870, 0.0039487
1: 0.0000533, 0.0077585, -0.0000491, 0.0078717, -0.0078185, 0.0078076
2: 0.0047207, 0.0162602, 0.0045512, 0.0164135, -0.0116927, 0.0117090
3: -0.0070771, 0.0016001, -0.0072046, 0.0017153, -0.0087925, 0.0088047
4: -0.0109076, -0.0016223, -0.0110251, -0.0015766, -0.0093310, 0.0094028
5: 0.0008740, 0.0095354, 0.0007467, 0.0096504, -0.0087764, 0.0087886
6: 0.0001117, 0.0126031, -0.0000600, 0.0126511, -0.0125394, 0.0126630
7: -0.0190997, -0.0002969, -0.0193494, -0.0000207, -0.0147118, 0.0148084
8: 0.9690681, 1.0229405, 0.9683527, 1.0237318, -0.0546637, 0.0545878
9: -0.0090891, 0.0067441, -0.0093217, 0.0069543, -0.0144480, 0.0144096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0326637, upper bound: 0.0297659
time: 0.77 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0326637, upper bound: 0.0336418
time: 1.01 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002706, 0.0042002, -0.0003058, 0.0042808, -0.0040094, 0.0039900
1: 0.0000533, 0.0077585, -0.0001113, 0.0078822, -0.0078289, 0.0078698
2: 0.0047207, 0.0162602, 0.0045355, 0.0165066, -0.0117859, 0.0117247
3: -0.0070771, 0.0016001, -0.0072164, 0.0017854, -0.0088625, 0.0088165
4: -0.0109076, -0.0016223, -0.0110360, -0.0015724, -0.0093352, 0.0094137
5: 0.0008740, 0.0095354, 0.0007349, 0.0097203, -0.0088463, 0.0088004
6: 0.0001117, 0.0126031, -0.0000758, 0.0126555, -0.0125438, 0.0126789
7: -0.0190997, -0.0002969, -0.0195011, 0.0000049, -0.0150594, 0.0152164
8: 0.9690681, 1.0229405, 0.9679180, 1.0238050, -0.0547369, 0.0550225
9: -0.0090891, 0.0067441, -0.0093432, 0.0070821, -0.0146790, 0.0147024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0325060, upper bound: 0.0294136
time: 0.95 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0325060, upper bound: 0.0335998
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002838, 0.0042071, -0.0002925, 0.0042740, -0.0040293, 0.0039680
1: -0.0000084, 0.0077691, -0.0000491, 0.0078717, -0.0078802, 0.0078182
2: 0.0047049, 0.0163526, 0.0045512, 0.0164135, -0.0117086, 0.0118014
3: -0.0070890, 0.0016696, -0.0072046, 0.0017153, -0.0088044, 0.0088741
4: -0.0109186, -0.0016180, -0.0110251, -0.0015766, -0.0093420, 0.0094071
5: 0.0008620, 0.0096047, 0.0007467, 0.0096504, -0.0087884, 0.0088580
6: 0.0000956, 0.0126076, -0.0000600, 0.0126511, -0.0125554, 0.0126675
7: -0.0192502, -0.0002710, -0.0193494, -0.0000207, -0.0151405, 0.0151141
8: 0.9686368, 1.0230147, 0.9683527, 1.0237318, -0.0550950, 0.0546620
9: -0.0091109, 0.0068708, -0.0093217, 0.0069543, -0.0147054, 0.0146706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0325896, upper bound: 0.0297167
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0325896, upper bound: 0.0336347
time: 1.30 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002838, 0.0042071, -0.0003058, 0.0042808, -0.0039999, 0.0039600
1: -0.0000084, 0.0077691, -0.0001113, 0.0078822, -0.0078906, 0.0078804
2: 0.0047049, 0.0163526, 0.0045355, 0.0165066, -0.0118017, 0.0118171
3: -0.0070890, 0.0016696, -0.0072164, 0.0017854, -0.0088744, 0.0088859
4: -0.0109186, -0.0016180, -0.0110360, -0.0015724, -0.0093462, 0.0094180
5: 0.0008620, 0.0096047, 0.0007349, 0.0097203, -0.0088582, 0.0088697
6: 0.0000956, 0.0126076, -0.0000758, 0.0126555, -0.0125599, 0.0126834
7: -0.0192502, -0.0002710, -0.0195011, 0.0000049, -0.0147314, 0.0148442
8: 0.9686368, 1.0230147, 0.9679180, 1.0238050, -0.0551682, 0.0550967
9: -0.0091109, 0.0068708, -0.0093432, 0.0070821, -0.0144736, 0.0143991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0325029, upper bound: 0.0294089
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0325029, upper bound: 0.0336140
time: 1.10 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003178, 0.0041808, -0.0002938, 0.0039692, -0.0037788, 0.0039566
1: -0.0001673, 0.0077290, -0.0000552, 0.0074046, -0.0075719, 0.0077841
2: 0.0047651, 0.0165906, 0.0052509, 0.0164226, -0.0116575, 0.0113397
3: -0.0070438, 0.0018485, -0.0066785, 0.0017222, -0.0087660, 0.0085270
4: -0.0108768, -0.0016343, -0.0105399, -0.0017653, -0.0091115, 0.0089056
5: 0.0009072, 0.0097833, 0.0012719, 0.0096572, -0.0087500, 0.0085115
6: 0.0001566, 0.0125905, 0.0006486, 0.0124529, -0.0122963, 0.0119419
7: -0.0196380, -0.0003691, -0.0193642, -0.0011607, -0.0146027, 0.0150375
8: 0.9675258, 1.0227336, 0.9683101, 1.0204656, -0.0529398, 0.0544235
9: -0.0090283, 0.0071973, -0.0083617, 0.0069668, -0.0145629, 0.0140773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0304820, upper bound: 0.0330580
time: 0.89 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304820, upper bound: 0.0344716
time: 1.41 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003346, 0.0041981, -0.0002938, 0.0039692, -0.0038127, 0.0039930
1: -0.0002460, 0.0077554, -0.0000552, 0.0074046, -0.0076506, 0.0078106
2: 0.0047254, 0.0167084, 0.0052509, 0.0164226, -0.0116972, 0.0114576
3: -0.0070736, 0.0019371, -0.0066785, 0.0017222, -0.0087958, 0.0086156
4: -0.0109043, -0.0016236, -0.0105399, -0.0017653, -0.0091390, 0.0089163
5: 0.0008775, 0.0098718, 0.0012719, 0.0096572, -0.0087798, 0.0085999
6: 0.0001164, 0.0126017, 0.0006486, 0.0124529, -0.0123365, 0.0119532
7: -0.0198300, -0.0003045, -0.0193642, -0.0011607, -0.0149650, 0.0153171
8: 0.9669758, 1.0229187, 0.9683101, 1.0204656, -0.0534898, 0.0546086
9: -0.0090827, 0.0073590, -0.0083617, 0.0069668, -0.0147984, 0.0143097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302838, upper bound: 0.0329730
time: 1.01 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302838, upper bound: 0.0344511
time: 0.96 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0003169, 0.0041125, -0.0003089, 0.0040527, -0.0038805, 0.0039296
1: -0.0001635, 0.0076241, -0.0001261, 0.0075326, -0.0076961, 0.0077502
2: 0.0049220, 0.0165848, 0.0050591, 0.0165288, -0.0116068, 0.0115257
3: -0.0069258, 0.0018442, -0.0068226, 0.0018021, -0.0087278, 0.0086668
4: -0.0107680, -0.0016766, -0.0106728, -0.0017136, -0.0090544, 0.0089962
5: 0.0010250, 0.0097790, 0.0011280, 0.0097369, -0.0087119, 0.0086510
6: 0.0003155, 0.0125460, 0.0004544, 0.0125072, -0.0121917, 0.0120916
7: -0.0196285, -0.0006248, -0.0195373, -0.0008483, -0.0151736, 0.0152081
8: 0.9675528, 1.0220008, 0.9678143, 1.0213606, -0.0538079, 0.0541865
9: -0.0088129, 0.0071894, -0.0086248, 0.0071126, -0.0146573, 0.0145187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0317671, upper bound: 0.0324248
time: 0.93 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0317671, upper bound: 0.0340014
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003346, 0.0041981, -0.0003081, 0.0039823, -0.0037987, 0.0039782
1: -0.0002460, 0.0077554, -0.0001223, 0.0074247, -0.0076707, 0.0078777
2: 0.0047254, 0.0167084, 0.0052206, 0.0165231, -0.0117977, 0.0114878
3: -0.0070736, 0.0019371, -0.0067012, 0.0017978, -0.0088714, 0.0086383
4: -0.0109043, -0.0016236, -0.0105608, -0.0017571, -0.0091472, 0.0089372
5: 0.0008775, 0.0098718, 0.0012492, 0.0097327, -0.0088552, 0.0086226
6: 0.0001164, 0.0126017, 0.0006180, 0.0124615, -0.0123450, 0.0119838
7: -0.0198300, -0.0003045, -0.0195280, -0.0011114, -0.0146686, 0.0150844
8: 0.9669758, 1.0229187, 0.9678407, 1.0206065, -0.0536307, 0.0550780
9: -0.0090827, 0.0073590, -0.0084032, 0.0071048, -0.0145792, 0.0141335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0302707, upper bound: 0.0329714
time: 1.11 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0302707, upper bound: 0.0344511
time: 0.89 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002889, 0.0043288, -0.0002946, 0.0040398, -0.0038514, 0.0041273
1: -0.0000322, 0.0079556, -0.0000590, 0.0075128, -0.0075450, 0.0080147
2: 0.0044255, 0.0163882, 0.0050888, 0.0164284, -0.0120028, 0.0112994
3: -0.0072991, 0.0016964, -0.0068003, 0.0017265, -0.0090256, 0.0084967
4: -0.0111123, -0.0015427, -0.0106523, -0.0017216, -0.0093907, 0.0091096
5: 0.0006524, 0.0096314, 0.0011502, 0.0096616, -0.0090092, 0.0084812
6: -0.0001872, 0.0126867, 0.0004845, 0.0124988, -0.0126860, 0.0122022
7: -0.0193082, 0.0001841, -0.0193736, -0.0008966, -0.0148421, 0.0159656
8: 0.9684706, 1.0243186, 0.9682832, 1.0212221, -0.0527515, 0.0560354
9: -0.0094941, 0.0069197, -0.0085841, 0.0069748, -0.0153873, 0.0141632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0259606, upper bound: 0.0306449
time: 1.24 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287147, upper bound: 0.0273690
time: 0.85 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002889, 0.0043288, -0.0003089, 0.0040527, -0.0038832, 0.0041696
1: -0.0000322, 0.0079556, -0.0001261, 0.0075326, -0.0075648, 0.0080817
2: 0.0044255, 0.0163882, 0.0050591, 0.0165288, -0.0121033, 0.0113291
3: -0.0072991, 0.0016964, -0.0068226, 0.0018021, -0.0091011, 0.0085190
4: -0.0111123, -0.0015427, -0.0106728, -0.0017136, -0.0093987, 0.0091301
5: 0.0006524, 0.0096314, 0.0011280, 0.0097369, -0.0090845, 0.0085035
6: -0.0001872, 0.0126867, 0.0004544, 0.0125072, -0.0126944, 0.0122323
7: -0.0193082, 0.0001841, -0.0195373, -0.0008483, -0.0151523, 0.0163977
8: 0.9684706, 1.0243186, 0.9678143, 1.0213606, -0.0528900, 0.0565042
9: -0.0094941, 0.0069197, -0.0086248, 0.0071126, -0.0156590, 0.0144244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292947, upper bound: 0.0306282
time: 0.97 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286815, upper bound: 0.0273405
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003020, 0.0043388, -0.0002946, 0.0040398, -0.0038816, 0.0041480
1: -0.0000938, 0.0079711, -0.0000590, 0.0075128, -0.0076066, 0.0080301
2: 0.0044024, 0.0164805, 0.0050888, 0.0164284, -0.0120259, 0.0113917
3: -0.0073165, 0.0017657, -0.0068003, 0.0017265, -0.0090430, 0.0085660
4: -0.0111283, -0.0015365, -0.0106523, -0.0017216, -0.0094068, 0.0091158
5: 0.0006350, 0.0097007, 0.0011502, 0.0096616, -0.0090265, 0.0085504
6: -0.0002106, 0.0126932, 0.0004845, 0.0124988, -0.0127094, 0.0122088
7: -0.0194585, 0.0002218, -0.0193736, -0.0008966, -0.0151903, 0.0161855
8: 0.9680399, 1.0244265, 0.9682832, 1.0212221, -0.0531822, 0.0561433
9: -0.0095258, 0.0070462, -0.0085841, 0.0069748, -0.0155725, 0.0143840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293216, upper bound: 0.0306468
time: 1.10 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287147, upper bound: 0.0273624
time: 0.85 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003020, 0.0043388, -0.0003089, 0.0040527, -0.0038714, 0.0041424
1: -0.0000938, 0.0079711, -0.0001261, 0.0075326, -0.0076264, 0.0080972
2: 0.0044024, 0.0164805, 0.0050591, 0.0165288, -0.0121264, 0.0114213
3: -0.0073165, 0.0017657, -0.0068226, 0.0018021, -0.0091185, 0.0085884
4: -0.0111283, -0.0015365, -0.0106728, -0.0017136, -0.0094148, 0.0091364
5: 0.0006350, 0.0097007, 0.0011280, 0.0097369, -0.0091019, 0.0085727
6: -0.0002106, 0.0126932, 0.0004544, 0.0125072, -0.0127178, 0.0122388
7: -0.0194585, 0.0002218, -0.0195373, -0.0008483, -0.0149061, 0.0160114
8: 0.9680399, 1.0244265, 0.9678143, 1.0213606, -0.0533208, 0.0566121
9: -0.0095258, 0.0070462, -0.0086248, 0.0071126, -0.0154021, 0.0142276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292947, upper bound: 0.0306464
time: 0.99 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286815, upper bound: 0.0273484
time: 1.07 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002938, 0.0039692, -0.0003178, 0.0041808, -0.0039566, 0.0037788
1: -0.0000552, 0.0074046, -0.0001673, 0.0077290, -0.0077841, 0.0075719
2: 0.0052509, 0.0164226, 0.0047651, 0.0165906, -0.0113397, 0.0116575
3: -0.0066785, 0.0017222, -0.0070438, 0.0018485, -0.0085270, 0.0087660
4: -0.0105399, -0.0017653, -0.0108768, -0.0016343, -0.0089056, 0.0091115
5: 0.0012719, 0.0096572, 0.0009072, 0.0097833, -0.0085115, 0.0087500
6: 0.0006486, 0.0124529, 0.0001566, 0.0125905, -0.0119419, 0.0122963
7: -0.0193642, -0.0011607, -0.0196380, -0.0003691, -0.0150375, 0.0146027
8: 0.9683101, 1.0204656, 0.9675258, 1.0227336, -0.0544235, 0.0529398
9: -0.0083617, 0.0069668, -0.0090283, 0.0071973, -0.0140773, 0.0145629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0330580, upper bound: 0.0306875
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0330580, upper bound: 0.0342143
time: 1.04 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002938, 0.0039692, -0.0003346, 0.0041981, -0.0039930, 0.0038127
1: -0.0000552, 0.0074046, -0.0002460, 0.0077554, -0.0078106, 0.0076506
2: 0.0052509, 0.0164226, 0.0047254, 0.0167084, -0.0114576, 0.0116972
3: -0.0066785, 0.0017222, -0.0070736, 0.0019371, -0.0086156, 0.0087958
4: -0.0105399, -0.0017653, -0.0109043, -0.0016236, -0.0089163, 0.0091390
5: 0.0012719, 0.0096572, 0.0008775, 0.0098718, -0.0085999, 0.0087798
6: 0.0006486, 0.0124529, 0.0001164, 0.0126017, -0.0119532, 0.0123365
7: -0.0193642, -0.0011607, -0.0198300, -0.0003045, -0.0153171, 0.0149650
8: 0.9683101, 1.0204656, 0.9669758, 1.0229187, -0.0546086, 0.0534898
9: -0.0083617, 0.0069668, -0.0090827, 0.0073590, -0.0143097, 0.0147984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0329730, upper bound: 0.0304994
time: 0.93 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0329730, upper bound: 0.0341862
time: 1.07 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003169, 0.0041125, -0.0039296, 0.0038805
1: -0.0001261, 0.0075326, -0.0001635, 0.0076241, -0.0077502, 0.0076961
2: 0.0050591, 0.0165288, 0.0049220, 0.0165848, -0.0115257, 0.0116068
3: -0.0068226, 0.0018021, -0.0069258, 0.0018442, -0.0086668, 0.0087278
4: -0.0106728, -0.0017136, -0.0107680, -0.0016766, -0.0089962, 0.0090544
5: 0.0011280, 0.0097369, 0.0010250, 0.0097790, -0.0086510, 0.0087119
6: 0.0004544, 0.0125072, 0.0003155, 0.0125460, -0.0120916, 0.0121917
7: -0.0195373, -0.0008483, -0.0196285, -0.0006248, -0.0152081, 0.0151736
8: 0.9678143, 1.0213606, 0.9675528, 1.0220008, -0.0541865, 0.0538079
9: -0.0086248, 0.0071126, -0.0088129, 0.0071894, -0.0145187, 0.0146573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0324248, upper bound: 0.0320046
time: 0.96 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0324248, upper bound: 0.0348398
time: 1.03 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003081, 0.0039823, -0.0003346, 0.0041981, -0.0039782, 0.0037987
1: -0.0001223, 0.0074247, -0.0002460, 0.0077554, -0.0078777, 0.0076707
2: 0.0052206, 0.0165231, 0.0047254, 0.0167084, -0.0114878, 0.0117977
3: -0.0067012, 0.0017978, -0.0070736, 0.0019371, -0.0086383, 0.0088714
4: -0.0105608, -0.0017571, -0.0109043, -0.0016236, -0.0089372, 0.0091472
5: 0.0012492, 0.0097327, 0.0008775, 0.0098718, -0.0086226, 0.0088552
6: 0.0006180, 0.0124615, 0.0001164, 0.0126017, -0.0119838, 0.0123450
7: -0.0195280, -0.0011114, -0.0198300, -0.0003045, -0.0150844, 0.0146686
8: 0.9678407, 1.0206065, 0.9669758, 1.0229187, -0.0550780, 0.0536307
9: -0.0084032, 0.0071048, -0.0090827, 0.0073590, -0.0141335, 0.0145792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0329714, upper bound: 0.0304988
time: 1.00 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0329714, upper bound: 0.0341885
time: 1.18 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0002889, 0.0043288, -0.0041273, 0.0038514
1: -0.0000590, 0.0075128, -0.0000322, 0.0079556, -0.0080147, 0.0075450
2: 0.0050888, 0.0164284, 0.0044255, 0.0163882, -0.0112994, 0.0120028
3: -0.0068003, 0.0017265, -0.0072991, 0.0016964, -0.0084967, 0.0090256
4: -0.0106523, -0.0017216, -0.0111123, -0.0015427, -0.0091096, 0.0093907
5: 0.0011502, 0.0096616, 0.0006524, 0.0096314, -0.0084812, 0.0090092
6: 0.0004845, 0.0124988, -0.0001872, 0.0126867, -0.0122022, 0.0126860
7: -0.0193736, -0.0008966, -0.0193082, 0.0001841, -0.0159656, 0.0148421
8: 0.9682832, 1.0212221, 0.9684706, 1.0243186, -0.0560354, 0.0527515
9: -0.0085841, 0.0069748, -0.0094941, 0.0069197, -0.0141632, 0.0153873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0306449, upper bound: 0.0293216
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0273690, upper bound: 0.0287147
time: 0.76 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0002889, 0.0043288, -0.0041696, 0.0038832
1: -0.0001261, 0.0075326, -0.0000322, 0.0079556, -0.0080817, 0.0075648
2: 0.0050591, 0.0165288, 0.0044255, 0.0163882, -0.0113291, 0.0121033
3: -0.0068226, 0.0018021, -0.0072991, 0.0016964, -0.0085190, 0.0091011
4: -0.0106728, -0.0017136, -0.0111123, -0.0015427, -0.0091301, 0.0093987
5: 0.0011280, 0.0097369, 0.0006524, 0.0096314, -0.0085035, 0.0090845
6: 0.0004544, 0.0125072, -0.0001872, 0.0126867, -0.0122323, 0.0126944
7: -0.0195373, -0.0008483, -0.0193082, 0.0001841, -0.0163977, 0.0151523
8: 0.9678143, 1.0213606, 0.9684706, 1.0243186, -0.0565042, 0.0528900
9: -0.0086248, 0.0071126, -0.0094941, 0.0069197, -0.0144244, 0.0156590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0306282, upper bound: 0.0292947
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0273405, upper bound: 0.0286815
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0003020, 0.0043388, -0.0041480, 0.0038816
1: -0.0000590, 0.0075128, -0.0000938, 0.0079711, -0.0080301, 0.0076066
2: 0.0050888, 0.0164284, 0.0044024, 0.0164805, -0.0113917, 0.0120259
3: -0.0068003, 0.0017265, -0.0073165, 0.0017657, -0.0085660, 0.0090430
4: -0.0106523, -0.0017216, -0.0111283, -0.0015365, -0.0091158, 0.0094068
5: 0.0011502, 0.0096616, 0.0006350, 0.0097007, -0.0085504, 0.0090265
6: 0.0004845, 0.0124988, -0.0002106, 0.0126932, -0.0122088, 0.0127094
7: -0.0193736, -0.0008966, -0.0194585, 0.0002218, -0.0161855, 0.0151903
8: 0.9682832, 1.0212221, 0.9680399, 1.0244265, -0.0561433, 0.0531822
9: -0.0085841, 0.0069748, -0.0095258, 0.0070462, -0.0143840, 0.0155725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0306468, upper bound: 0.0293216
time: 1.18 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0273624, upper bound: 0.0287147
time: 1.07 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0003020, 0.0043388, -0.0041424, 0.0038714
1: -0.0001261, 0.0075326, -0.0000938, 0.0079711, -0.0080972, 0.0076264
2: 0.0050591, 0.0165288, 0.0044024, 0.0164805, -0.0114213, 0.0121264
3: -0.0068226, 0.0018021, -0.0073165, 0.0017657, -0.0085884, 0.0091185
4: -0.0106728, -0.0017136, -0.0111283, -0.0015365, -0.0091364, 0.0094148
5: 0.0011280, 0.0097369, 0.0006350, 0.0097007, -0.0085727, 0.0091019
6: 0.0004544, 0.0125072, -0.0002106, 0.0126932, -0.0122388, 0.0127178
7: -0.0195373, -0.0008483, -0.0194585, 0.0002218, -0.0160114, 0.0149061
8: 0.9678143, 1.0213606, 0.9680399, 1.0244265, -0.0566121, 0.0533208
9: -0.0086248, 0.0071126, -0.0095258, 0.0070462, -0.0142276, 0.0154021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0306282, upper bound: 0.0292947
time: 0.97 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0273405, upper bound: 0.0286815
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005796, 0.0041177, -0.0005796, 0.0041177, -0.0041142, 0.0041142
1: -0.0003968, 0.0076321, -0.0003968, 0.0076321, -0.0080289, 0.0080289
2: 0.0049100, 0.0173714, 0.0049100, 0.0173714, -0.0124613, 0.0124613
3: -0.0069347, 0.0022288, -0.0069347, 0.0022288, -0.0091635, 0.0091635
4: -0.0107763, -0.0016418, -0.0107763, -0.0016418, -0.0091345, 0.0091345
5: 0.0010112, 0.0100184, 0.0010112, 0.0100184, -0.0090072, 0.0090072
6: 0.0003034, 0.0125494, 0.0003034, 0.0125494, -0.0122460, 0.0122460
7: -0.0203213, -0.0006053, -0.0203213, -0.0006053, -0.0152554, 0.0152554
8: 0.9659874, 1.0220567, 0.9659874, 1.0220567, -0.0560693, 0.0560693
9: -0.0088434, 0.0075755, -0.0088434, 0.0075755, -0.0147674, 0.0147674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339483, upper bound: 0.0308488
time: 0.91 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365657, upper bound: 0.0361995
time: 1.37 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0019739, 0.0041329, -0.0005796, 0.0041177, -0.0054794, 0.0041512
1: -0.0008638, 0.0076555, -0.0003968, 0.0076321, -0.0084960, 0.0080523
2: 0.0048751, 0.0207585, 0.0049100, 0.0173714, -0.0124963, 0.0150431
3: -0.0069611, 0.0035267, -0.0069347, 0.0022288, -0.0091898, 0.0104034
4: -0.0108005, 0.0011510, -0.0107763, -0.0016418, -0.0091587, 0.0119273
5: 0.0009551, 0.0104198, 0.0010112, 0.0100184, -0.0090632, 0.0094086
6: 0.0002680, 0.0125593, 0.0003034, 0.0125494, -0.0122814, 0.0122559
7: -0.0222936, -0.0005483, -0.0203213, -0.0006053, -0.0170498, 0.0156198
8: 0.9630781, 1.0222199, 0.9659874, 1.0220567, -0.0589786, 0.0562325
9: -0.0089799, 0.0080311, -0.0088434, 0.0075755, -0.0155624, 0.0151109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0308796, upper bound: 0.0338428
time: 1.55 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365657, upper bound: 0.0362339
time: 0.99 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005796, 0.0041177, -0.0019739, 0.0041329, -0.0041512, 0.0054794
1: -0.0003968, 0.0076321, -0.0008638, 0.0076555, -0.0080523, 0.0084960
2: 0.0049100, 0.0173714, 0.0048751, 0.0207585, -0.0150431, 0.0124963
3: -0.0069347, 0.0022288, -0.0069611, 0.0035267, -0.0104034, 0.0091898
4: -0.0107763, -0.0016418, -0.0108005, 0.0011510, -0.0119273, 0.0091587
5: 0.0010112, 0.0100184, 0.0009551, 0.0104198, -0.0094086, 0.0090632
6: 0.0003034, 0.0125494, 0.0002680, 0.0125593, -0.0122559, 0.0122814
7: -0.0203213, -0.0006053, -0.0222936, -0.0005483, -0.0156198, 0.0170498
8: 0.9659874, 1.0220567, 0.9630781, 1.0222199, -0.0562325, 0.0589786
9: -0.0088434, 0.0075755, -0.0089799, 0.0080311, -0.0151109, 0.0155624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339483, upper bound: 0.0308488
time: 1.05 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365651, upper bound: 0.0361991
time: 1.07 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0019739, 0.0041329, -0.0019739, 0.0041329, -0.0054723, 0.0054723
1: -0.0008638, 0.0076555, -0.0008638, 0.0076555, -0.0085193, 0.0085193
2: 0.0048751, 0.0207585, 0.0048751, 0.0207585, -0.0148427, 0.0148427
3: -0.0069611, 0.0035267, -0.0069611, 0.0035267, -0.0102800, 0.0102800
4: -0.0108005, 0.0011510, -0.0108005, 0.0011510, -0.0119515, 0.0119515
5: 0.0009551, 0.0104198, 0.0009551, 0.0104198, -0.0094647, 0.0094647
6: 0.0002680, 0.0125593, 0.0002680, 0.0125593, -0.0122913, 0.0122913
7: -0.0222936, -0.0005483, -0.0222936, -0.0005483, -0.0166415, 0.0166415
8: 0.9630781, 1.0222199, 0.9630781, 1.0222199, -0.0591418, 0.0591418
9: -0.0089799, 0.0080311, -0.0089799, 0.0080311, -0.0153927, 0.0153927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339483, upper bound: 0.0308488
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0365651, upper bound: 0.0362339
time: 1.12 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0103240, 0.0040144, -0.0103240, 0.0040144, -0.0137004, 0.0137004
1: -0.0025680, 0.0074739, -0.0025680, 0.0074739, -0.0100419, 0.0100419
2: 0.0051471, 0.0408788, 0.0051471, 0.0408788, -0.0344752, 0.0344752
3: -0.0067565, 0.0111193, -0.0067565, 0.0111193, -0.0167660, 0.0167660
4: -0.0106119, 0.0195539, -0.0106119, 0.0195539, -0.0301658, 0.0301658
5: 0.0009396, 0.0118735, 0.0009396, 0.0118735, -0.0109339, 0.0109339
6: 0.0005434, 0.0124823, 0.0005434, 0.0124823, -0.0119388, 0.0119388
7: -0.0336609, -0.0009915, -0.0336609, -0.0009915, -0.0273102, 0.0273102
8: 0.9515460, 1.0209502, 0.9515460, 1.0209502, -0.0694042, 0.0694042
9: -0.0093437, 0.0096484, -0.0093437, 0.0096484, -0.0189920, 0.0189920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0251539, upper bound: 0.0298822
time: 0.98 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349045, upper bound: 0.0347726
time: 1.19 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0103240, 0.0040144, -0.0117704, 0.0040339, -0.0137392, 0.0151544
1: -0.0025680, 0.0074739, -0.0028126, 0.0075037, -0.0100717, 0.0102865
2: 0.0051471, 0.0408788, 0.0051024, 0.0444233, -0.0382967, 0.0346959
3: -0.0067565, 0.0111193, -0.0067901, 0.0124899, -0.0183423, 0.0169320
4: -0.0106119, 0.0195539, -0.0106428, 0.0227793, -0.0333911, 0.0301968
5: 0.0009396, 0.0118735, 0.0008732, 0.0120954, -0.0111559, 0.0110002
6: 0.0005434, 0.0124823, 0.0004982, 0.0124949, -0.0119515, 0.0119841
7: -0.0336609, -0.0009915, -0.0357505, -0.0009187, -0.0276698, 0.0296836
8: 0.9515460, 1.0209502, 0.9491154, 1.0211587, -0.0696127, 0.0718348
9: -0.0093437, 0.0096484, -0.0096065, 0.0099348, -0.0192784, 0.0192549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299324, upper bound: 0.0251761
time: 0.78 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A1_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0349045, upper bound: 0.0347726
time: 0.91 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0117704, 0.0040339, -0.0103240, 0.0040144, -0.0151544, 0.0137392
1: -0.0028126, 0.0075037, -0.0025680, 0.0074739, -0.0102865, 0.0100717
2: 0.0051024, 0.0444233, 0.0051471, 0.0408788, -0.0346959, 0.0382967
3: -0.0067901, 0.0124899, -0.0067565, 0.0111193, -0.0169320, 0.0183423
4: -0.0106428, 0.0227793, -0.0106119, 0.0195539, -0.0301968, 0.0333911
5: 0.0008732, 0.0120954, 0.0009396, 0.0118735, -0.0110002, 0.0111559
6: 0.0004982, 0.0124949, 0.0005434, 0.0124823, -0.0119841, 0.0119515
7: -0.0357505, -0.0009187, -0.0336609, -0.0009915, -0.0296836, 0.0276698
8: 0.9491154, 1.0211587, 0.9515460, 1.0209502, -0.0718348, 0.0696127
9: -0.0096065, 0.0099348, -0.0093437, 0.0096484, -0.0192549, 0.0192784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0251365, upper bound: 0.0298811
time: 0.97 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348993, upper bound: 0.0347564
time: 1.05 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0117704, 0.0040339, -0.0117704, 0.0040339, -0.0151590, 0.0151590
1: -0.0028126, 0.0075037, -0.0028126, 0.0075037, -0.0103163, 0.0103163
2: 0.0051024, 0.0444233, 0.0051024, 0.0444233, -0.0381876, 0.0381876
3: -0.0067901, 0.0124899, -0.0067901, 0.0124899, -0.0182267, 0.0182267
4: -0.0106428, 0.0227793, -0.0106428, 0.0227793, -0.0334221, 0.0334221
5: 0.0008732, 0.0120954, 0.0008732, 0.0120954, -0.0112222, 0.0112222
6: 0.0004982, 0.0124949, 0.0004982, 0.0124949, -0.0119968, 0.0119968
7: -0.0357505, -0.0009187, -0.0357505, -0.0009187, -0.0293793, 0.0293793
8: 0.9491154, 1.0211587, 0.9491154, 1.0211587, -0.0720433, 0.0720433
9: -0.0096065, 0.0099348, -0.0096065, 0.0099348, -0.0195413, 0.0195413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0251365, upper bound: 0.0298811
time: 1.45 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0348993, upper bound: 0.0347564
time: 1.24 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.39 seconds
IS_B1_A1_A2_B1_A1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0333796, upper bound: 0.0309229
IS_B1_A1_A2_B1_A1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0333796, upper bound: 0.0309229
IS_B1_A1_A2_B1_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0332749, upper bound: 0.0306599
IS_B1_A1_A2_B1_A1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0332749, upper bound: 0.0341322
IS_B1_A1_A2_B1_A1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0333246, upper bound: 0.0308986
IS_B1_A1_A2_B1_A1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0333246, upper bound: 0.0341345
IS_B1_A1_A2_B1_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0332677, upper bound: 0.0306572
IS_B1_A1_A2_B1_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0332677, upper bound: 0.0341326
IS_B1_A1_A2_B1_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0328644, upper bound: 0.0306440
IS_B1_A1_A2_B1_A2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0328644, upper bound: 0.0338803
IS_B1_A1_A2_B1_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0327310, upper bound: 0.0303682
IS_B1_A1_A2_B1_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0327310, upper bound: 0.0338574
IS_B1_A1_A2_B1_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0328029, upper bound: 0.0305658
IS_B1_A1_A2_B1_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0328029, upper bound: 0.0338699
IS_B1_A1_A2_B1_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0327306, upper bound: 0.0303608
IS_B1_A1_A2_B1_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0327306, upper bound: 0.0338687
IS_B1_A1_A2_B2_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0321519, upper bound: 0.0313857
IS_B1_A1_A2_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0321519, upper bound: 0.0344684
IS_B1_A1_A2_B2_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0320508, upper bound: 0.0313017
IS_B1_A1_A2_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0320508, upper bound: 0.0344579
IS_B1_A1_A2_B2_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0320476, upper bound: 0.0312512
IS_B1_A1_A2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0320476, upper bound: 0.0344467
IS_B1_A1_A2_B2_A1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0320076, upper bound: 0.0312332
IS_B1_A1_A2_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0320076, upper bound: 0.0344579
IS_B1_A1_A2_B2_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0326637, upper bound: 0.0297659
IS_B1_A1_A2_B2_A2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0326637, upper bound: 0.0336418
IS_B1_A1_A2_B2_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0325060, upper bound: 0.0294136
IS_B1_A1_A2_B2_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0325060, upper bound: 0.0335998
IS_B1_A1_A2_B2_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0325896, upper bound: 0.0297167
IS_B1_A1_A2_B2_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0325896, upper bound: 0.0336347
IS_B1_A1_A2_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0325029, upper bound: 0.0294089
IS_B1_A1_A2_B2_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0325029, upper bound: 0.0336140
IS_B1_A2_B2_A1_B1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0304820, upper bound: 0.0330580
IS_B1_A2_B2_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0304820, upper bound: 0.0344716
IS_B1_A2_B2_A1_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0302838, upper bound: 0.0329730
IS_B1_A2_B2_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0302838, upper bound: 0.0344511
IS_B1_A2_B2_A1_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0317671, upper bound: 0.0324248
IS_B1_A2_B2_A1_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0317671, upper bound: 0.0340014
IS_B1_A2_B2_A1_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0302707, upper bound: 0.0329714
IS_B1_A2_B2_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0302707, upper bound: 0.0344511
IS_B1_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0259606, upper bound: 0.0306449
IS_B1_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0287147, upper bound: 0.0273690
IS_B1_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0292947, upper bound: 0.0306282
IS_B1_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0286815, upper bound: 0.0273405
IS_B1_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0293216, upper bound: 0.0306468
IS_B1_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0287147, upper bound: 0.0273624
IS_B1_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0292947, upper bound: 0.0306464
IS_B1_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0286815, upper bound: 0.0273484
IS_B2_A2_A1_B1_A1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0330580, upper bound: 0.0306875
IS_B2_A2_A1_B1_A1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0330580, upper bound: 0.0342143
IS_B2_A2_A1_B1_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0329730, upper bound: 0.0304994
IS_B2_A2_A1_B1_A1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0329730, upper bound: 0.0341862
IS_B2_A2_A1_B1_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0324248, upper bound: 0.0320046
IS_B2_A2_A1_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0324248, upper bound: 0.0348398
IS_B2_A2_A1_B1_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0329714, upper bound: 0.0304988
IS_B2_A2_A1_B1_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0329714, upper bound: 0.0341885
IS_B2_A2_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0306449, upper bound: 0.0293216
IS_B2_A2_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0273690, upper bound: 0.0287147
IS_B2_A2_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0306282, upper bound: 0.0292947
IS_B2_A2_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0273405, upper bound: 0.0286815
IS_B2_A2_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0306468, upper bound: 0.0293216
IS_B2_A2_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0273624, upper bound: 0.0287147
IS_B2_A2_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0306282, upper bound: 0.0292947
IS_B2_A2_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0273405, upper bound: 0.0286815
IS_B2_A2_A2_B2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0339483, upper bound: 0.0308488
IS_B2_A2_A2_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0365657, upper bound: 0.0361995
IS_B2_A2_A2_B2_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0308796, upper bound: 0.0338428
IS_B2_A2_A2_B2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0365657, upper bound: 0.0362339
IS_B2_A2_A2_B2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0339483, upper bound: 0.0308488
IS_B2_A2_A2_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0365651, upper bound: 0.0361991
IS_B2_A2_A2_B2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0339483, upper bound: 0.0308488
IS_B2_A2_A2_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0365651, upper bound: 0.0362339
IS_B2_A2_A2_B2_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0251539, upper bound: 0.0298822
IS_B2_A2_A2_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0349045, upper bound: 0.0347726
IS_B2_A2_A2_B2_B2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0299324, upper bound: 0.0251761
IS_B2_A2_A2_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0349045, upper bound: 0.0347726
IS_B2_A2_A2_B2_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0251365, upper bound: 0.0298811
IS_B2_A2_A2_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0348993, upper bound: 0.0347564
IS_B2_A2_A2_B2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0251365, upper bound: 0.0298811
IS_B2_A2_A2_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 8, lower bound: -0.0348993, upper bound: 0.0347564

## BFS IS instance: IS_B1_A1_A2_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0002946, 0.0040398, -0.0002706, 0.0042002, -0.0040003, 0.0038169
1: -0.0000590, 0.0075128, 0.0000533, 0.0077585, -0.0078176, 0.0074595
2: 0.0050888, 0.0164284, 0.0047207, 0.0162602, -0.0111714, 0.0117076
3: -0.0068003, 0.0017265, -0.0070771, 0.0016001, -0.0084004, 0.0088037
4: -0.0106523, -0.0017216, -0.0109076, -0.0016223, -0.0090299, 0.0091860
5: 0.0011502, 0.0096616, 0.0008740, 0.0095354, -0.0083851, 0.0087876
6: 0.0004845, 0.0124988, 0.0001117, 0.0126031, -0.0121186, 0.0123871
7: -0.0193736, -0.0008966, -0.0190997, -0.0002969, -0.0155102, 0.0144471
8: 0.9682832, 1.0212221, 0.9690681, 1.0229405, -0.0546573, 0.0521540
9: -0.0085841, 0.0069748, -0.0090891, 0.0067441, -0.0139766, 0.0150039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0243259, upper bound: 0.0312362
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0321519, upper bound: 0.0344684
time: 0.90 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0321519, upper bound: 0.0344684
time: 0.91 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0003089, 0.0040527, -0.0002706, 0.0042002, -0.0040426, 0.0038509
1: -0.0001261, 0.0075326, 0.0000533, 0.0077585, -0.0078846, 0.0074793
2: 0.0050591, 0.0165288, 0.0047207, 0.0162602, -0.0112011, 0.0118081
3: -0.0068226, 0.0018021, -0.0070771, 0.0016001, -0.0084228, 0.0088792
4: -0.0106728, -0.0017136, -0.0109076, -0.0016223, -0.0090505, 0.0091940
5: 0.0011280, 0.0097369, 0.0008740, 0.0095354, -0.0084074, 0.0088630
6: 0.0004544, 0.0125072, 0.0001117, 0.0126031, -0.0121486, 0.0123955
7: -0.0195373, -0.0008483, -0.0190997, -0.0002969, -0.0159423, 0.0148026
8: 0.9678143, 1.0213606, 0.9690681, 1.0229405, -0.0551262, 0.0522925
9: -0.0086248, 0.0071126, -0.0090891, 0.0067441, -0.0142759, 0.0152756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.33 + 598.05 = 601.38 seconds
