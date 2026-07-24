## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0013


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9893919, 0.9935765, 0.9893919, 0.9935765, -0.0030069, 0.0030069)
1: (-0.0039072, -0.0028645, -0.0039072, -0.0028645, -0.0007492, 0.0007492)
2: (0.0051264, 0.0106521, 0.0051264, 0.0106521, -0.0039706, 0.0039706)
3: (-0.0061215, -0.0036065, -0.0061215, -0.0036065, -0.0018072, 0.0018072)
4: (0.0015201, 0.0025896, 0.0015201, 0.0025896, -0.0007685, 0.0007685)
5: (0.0054072, 0.0123570, 0.0054072, 0.0123570, -0.0049939, 0.0049939)
6: (-0.0015955, 0.0001684, -0.0015955, 0.0001684, -0.0012675, 0.0012675)
7: (-0.0072657, -0.0027018, -0.0072657, -0.0027018, -0.0032794, 0.0032794)
8: (-0.0033851, -0.0009850, -0.0033851, -0.0009850, -0.0017246, 0.0017246)
9: (-0.0007217, 0.0020613, -0.0007217, 0.0020613, -0.0019998, 0.0019998)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.67 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0016691, upper bound: 0.0016691

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016250, upper bound: 0.0015646
time: 0.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016250, upper bound: 0.0016250
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -0.0016250, upper bound: 0.0015646
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -0.0016250, upper bound: 0.0016250

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9894740, 0.9933803, 0.9893929, 0.9935159, -0.0028591, 0.0028155
1: -0.0038868, -0.0029134, -0.0039070, -0.0028796, -0.0007124, 0.0007016
2: 0.0053856, 0.0105438, 0.0052064, 0.0106508, -0.0037179, 0.0037754
3: -0.0060722, -0.0037244, -0.0061209, -0.0036429, -0.0017184, 0.0016922
4: 0.0015703, 0.0025686, 0.0015356, 0.0025893, -0.0007196, 0.0007307
5: 0.0057331, 0.0122208, 0.0055077, 0.0123553, -0.0046761, 0.0047485
6: -0.0015609, 0.0000857, -0.0015951, 0.0001429, -0.0012052, 0.0011868
7: -0.0071762, -0.0029158, -0.0072646, -0.0027679, -0.0031183, 0.0030707
8: -0.0033381, -0.0010976, -0.0033845, -0.0010197, -0.0016399, 0.0016149
9: -0.0005912, 0.0020068, -0.0006814, 0.0020607, -0.0018725, 0.0019015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0014961
time: 0.79 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015027
time: 0.79 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9893942, 0.9934682, 0.9893926, 0.9935428, -0.0029229, 0.0028453
1: -0.0039067, -0.0028915, -0.0039070, -0.0028729, -0.0007283, 0.0007090
2: 0.0052695, 0.0106492, 0.0051711, 0.0106512, -0.0037571, 0.0038597
3: -0.0061202, -0.0036716, -0.0061211, -0.0036268, -0.0017567, 0.0017101
4: 0.0015478, 0.0025890, 0.0015287, 0.0025894, -0.0007272, 0.0007470
5: 0.0055871, 0.0123534, 0.0054633, 0.0123559, -0.0047255, 0.0048544
6: -0.0015946, 0.0001228, -0.0015952, 0.0001542, -0.0012321, 0.0011994
7: -0.0072633, -0.0028200, -0.0072649, -0.0027387, -0.0031878, 0.0031032
8: -0.0033839, -0.0010472, -0.0033847, -0.0010044, -0.0016765, 0.0016319
9: -0.0006496, 0.0020599, -0.0006992, 0.0020609, -0.0018923, 0.0019439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015590
time: 0.84 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015638
time: 0.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.09 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0014961
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015027
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015590
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015638

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.9894834, 0.9933004, 0.9893944, 0.9935035, -0.0028277, 0.0027163
1: -0.0038844, -0.0029333, -0.0039066, -0.0028827, -0.0007046, 0.0006768
2: 0.0054912, 0.0105315, 0.0052229, 0.0106489, -0.0035869, 0.0037339
3: -0.0060666, -0.0037725, -0.0061200, -0.0036503, -0.0016995, 0.0016326
4: 0.0015907, 0.0025662, 0.0015388, 0.0025890, -0.0006942, 0.0007227
5: 0.0058659, 0.0122053, 0.0055285, 0.0123529, -0.0045113, 0.0046963
6: -0.0015570, 0.0000520, -0.0015945, 0.0001376, -0.0011920, 0.0011450
7: -0.0071660, -0.0030031, -0.0072630, -0.0027815, -0.0030840, 0.0029625
8: -0.0033327, -0.0011434, -0.0033837, -0.0010269, -0.0016218, 0.0015580
9: -0.0005380, 0.0020006, -0.0006731, 0.0020597, -0.0018065, 0.0018806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0013918
time: 0.76 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014077
time: 0.81 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.9893860, 0.9932795, 0.9893982, 0.9934617, -0.0028939, 0.0027937
1: -0.0039087, -0.0029385, -0.0039056, -0.0028931, -0.0007211, 0.0006961
2: 0.0055186, 0.0106600, 0.0052779, 0.0106439, -0.0036891, 0.0038213
3: -0.0061251, -0.0037850, -0.0061177, -0.0036754, -0.0017393, 0.0016791
4: 0.0015960, 0.0025911, 0.0015494, 0.0025880, -0.0007140, 0.0007396
5: 0.0059005, 0.0123669, 0.0055977, 0.0123466, -0.0046399, 0.0048062
6: -0.0015980, 0.0000432, -0.0015929, 0.0001201, -0.0012199, 0.0011777
7: -0.0072722, -0.0030258, -0.0072589, -0.0028270, -0.0031562, 0.0030470
8: -0.0033885, -0.0011554, -0.0033815, -0.0010508, -0.0016598, 0.0016024
9: -0.0005241, 0.0020653, -0.0006454, 0.0020572, -0.0018580, 0.0019246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014199
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014199
time: 0.76 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.9894034, 0.9933904, 0.9893941, 0.9935302, -0.0028893, 0.0027441
1: -0.0039043, -0.0029109, -0.0039067, -0.0028761, -0.0007199, 0.0006837
2: 0.0053722, 0.0106369, 0.0051876, 0.0106492, -0.0036235, 0.0038153
3: -0.0061146, -0.0037183, -0.0061202, -0.0036343, -0.0017366, 0.0016493
4: 0.0015677, 0.0025866, 0.0015319, 0.0025890, -0.0007013, 0.0007384
5: 0.0057163, 0.0123379, 0.0054841, 0.0123534, -0.0045574, 0.0047986
6: -0.0015907, 0.0000900, -0.0015946, 0.0001489, -0.0012179, 0.0011567
7: -0.0072531, -0.0029048, -0.0072633, -0.0027523, -0.0031512, 0.0029928
8: -0.0033785, -0.0010918, -0.0033839, -0.0010116, -0.0016572, 0.0015739
9: -0.0005979, 0.0020537, -0.0006909, 0.0020599, -0.0018250, 0.0019216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014506
time: 0.76 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014756
time: 0.80 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.9893115, 0.9933687, 0.9893979, 0.9934888, -0.0029413, 0.0028187
1: -0.0039273, -0.0029163, -0.0039057, -0.0028864, -0.0007329, 0.0007023
2: 0.0054008, 0.0107584, 0.0052422, 0.0106442, -0.0037220, 0.0038840
3: -0.0061699, -0.0037313, -0.0061179, -0.0036591, -0.0017678, 0.0016941
4: 0.0015732, 0.0026102, 0.0015425, 0.0025881, -0.0007204, 0.0007517
5: 0.0057522, 0.0124907, 0.0055528, 0.0123471, -0.0046813, 0.0048850
6: -0.0016294, 0.0000808, -0.0015930, 0.0001315, -0.0012399, 0.0011882
7: -0.0073535, -0.0029284, -0.0072592, -0.0027974, -0.0032079, 0.0030742
8: -0.0034313, -0.0011042, -0.0033817, -0.0010353, -0.0016870, 0.0016167
9: -0.0005835, 0.0021149, -0.0006634, 0.0020574, -0.0018746, 0.0019562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014893
time: 0.89 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014893
time: 0.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.13 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0013918
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014077
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014199
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014199
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014506
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014756
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014893
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014893

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9894857, 0.9932311, 0.9893944, 0.9935035, -0.0028183, 0.0026489
1: -0.0038838, -0.0029506, -0.0039066, -0.0028827, -0.0007022, 0.0006600
2: 0.0055825, 0.0105283, 0.0052229, 0.0106489, -0.0034978, 0.0037215
3: -0.0060651, -0.0038140, -0.0061200, -0.0036503, -0.0016939, 0.0015921
4: 0.0016084, 0.0025656, 0.0015388, 0.0025890, -0.0006770, 0.0007203
5: 0.0059807, 0.0122012, 0.0055285, 0.0123529, -0.0043994, 0.0046807
6: -0.0015560, 0.0000229, -0.0015945, 0.0001376, -0.0011880, 0.0011166
7: -0.0071634, -0.0030785, -0.0072630, -0.0027815, -0.0030737, 0.0028890
8: -0.0033313, -0.0011831, -0.0033837, -0.0010269, -0.0016164, 0.0015193
9: -0.0004920, 0.0019990, -0.0006731, 0.0020597, -0.0017617, 0.0018743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0013911
time: 0.80 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0013911
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9891587, 0.9931157, 0.9893975, 0.9934269, -0.0031321, 0.0027693
1: -0.0039653, -0.0029794, -0.0039058, -0.0029018, -0.0007804, 0.0006900
2: 0.0057350, 0.0109600, 0.0053241, 0.0106447, -0.0036568, 0.0041359
3: -0.0062616, -0.0038835, -0.0061181, -0.0036964, -0.0018825, 0.0016644
4: 0.0016379, 0.0026492, 0.0015584, 0.0025882, -0.0007078, 0.0008005
5: 0.0061726, 0.0127442, 0.0056558, 0.0123477, -0.0045993, 0.0052019
6: -0.0016938, -0.0000258, -0.0015932, 0.0001053, -0.0013203, 0.0011673
7: -0.0075200, -0.0032045, -0.0072596, -0.0028651, -0.0034160, 0.0030203
8: -0.0035188, -0.0012494, -0.0033819, -0.0010709, -0.0017964, 0.0015883
9: -0.0004152, 0.0022164, -0.0006221, 0.0020576, -0.0018417, 0.0020831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014591, upper bound: 0.0013375
time: 0.96 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0013803
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9893860, 0.9932795, 0.9894006, 0.9933869, -0.0028087, 0.0027839
1: -0.0039087, -0.0029385, -0.0039050, -0.0029118, -0.0006999, 0.0006937
2: 0.0055186, 0.0106600, 0.0053768, 0.0106407, -0.0036761, 0.0037089
3: -0.0061251, -0.0037850, -0.0061163, -0.0037204, -0.0016881, 0.0016732
4: 0.0015960, 0.0025911, 0.0015686, 0.0025874, -0.0007115, 0.0007179
5: 0.0059005, 0.0123669, 0.0057220, 0.0123426, -0.0046235, 0.0046648
6: -0.0015980, 0.0000432, -0.0015919, 0.0000885, -0.0011840, 0.0011735
7: -0.0072722, -0.0030258, -0.0072562, -0.0029086, -0.0030633, 0.0030362
8: -0.0033885, -0.0011554, -0.0033801, -0.0010938, -0.0016110, 0.0015967
9: -0.0005241, 0.0020653, -0.0005956, 0.0020556, -0.0018515, 0.0018680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0013938
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014199
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9893893, 0.9932062, 0.9890689, 0.9932874, -0.0029249, 0.0030715
1: -0.0039078, -0.0029568, -0.0039877, -0.0029366, -0.0007288, 0.0007653
2: 0.0056154, 0.0106555, 0.0055082, 0.0110787, -0.0040558, 0.0038624
3: -0.0061231, -0.0038290, -0.0063157, -0.0037802, -0.0017580, 0.0018460
4: 0.0016147, 0.0025902, 0.0015940, 0.0026721, -0.0007850, 0.0007476
5: 0.0060222, 0.0123613, 0.0058873, 0.0128935, -0.0051012, 0.0048578
6: -0.0015966, 0.0000123, -0.0017317, 0.0000466, -0.0012330, 0.0012947
7: -0.0072685, -0.0031057, -0.0076180, -0.0030171, -0.0031901, 0.0033499
8: -0.0033866, -0.0011974, -0.0035704, -0.0011508, -0.0016776, 0.0017617
9: -0.0004754, 0.0020631, -0.0005294, 0.0022762, -0.0020427, 0.0019453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0013496
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0013915
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9894058, 0.9933192, 0.9893941, 0.9935302, -0.0028796, 0.0026773
1: -0.0039037, -0.0029286, -0.0039067, -0.0028761, -0.0007175, 0.0006671
2: 0.0054663, 0.0106338, 0.0051876, 0.0106492, -0.0035353, 0.0038025
3: -0.0061131, -0.0037611, -0.0061202, -0.0036343, -0.0017307, 0.0016091
4: 0.0015859, 0.0025860, 0.0015319, 0.0025890, -0.0006843, 0.0007360
5: 0.0058346, 0.0123339, 0.0054841, 0.0123534, -0.0044465, 0.0047825
6: -0.0015896, 0.0000600, -0.0015946, 0.0001489, -0.0012139, 0.0011286
7: -0.0072505, -0.0029825, -0.0072633, -0.0027523, -0.0031406, 0.0029200
8: -0.0033771, -0.0011326, -0.0033839, -0.0010116, -0.0016516, 0.0015356
9: -0.0005505, 0.0020521, -0.0006909, 0.0020599, -0.0017806, 0.0019151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014503
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014503
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9890760, 0.9932151, 0.9893973, 0.9934551, -0.0031850, 0.0028020
1: -0.0039859, -0.0029546, -0.0039059, -0.0028948, -0.0007936, 0.0006982
2: 0.0056037, 0.0110692, 0.0052869, 0.0106451, -0.0037000, 0.0042058
3: -0.0063114, -0.0038237, -0.0061183, -0.0036795, -0.0019143, 0.0016841
4: 0.0016125, 0.0026703, 0.0015511, 0.0025882, -0.0007161, 0.0008140
5: 0.0060074, 0.0128816, 0.0056089, 0.0123482, -0.0046536, 0.0052897
6: -0.0017287, 0.0000161, -0.0015933, 0.0001172, -0.0013426, 0.0011811
7: -0.0076102, -0.0030960, -0.0072599, -0.0028343, -0.0034737, 0.0030560
8: -0.0035663, -0.0011923, -0.0033821, -0.0010547, -0.0018268, 0.0016071
9: -0.0004813, 0.0022714, -0.0006409, 0.0020578, -0.0018635, 0.0021182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_A2_A1

### Relational analysis result of IS_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014128, upper bound: 0.0014460
time: 0.84 seconds

## Relational analysis of IS_A2_A1_A2_A2

### Relational analysis result of IS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0014460
time: 0.88 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9893115, 0.9933687, 0.9894004, 0.9934136, -0.0028535, 0.0028090
1: -0.0039273, -0.0029163, -0.0039051, -0.0029051, -0.0007110, 0.0006999
2: 0.0054008, 0.0107584, 0.0053416, 0.0106410, -0.0037093, 0.0037680
3: -0.0061699, -0.0037313, -0.0061164, -0.0037044, -0.0017150, 0.0016883
4: 0.0015732, 0.0026102, 0.0015617, 0.0025874, -0.0007179, 0.0007293
5: 0.0057522, 0.0124907, 0.0056778, 0.0123430, -0.0046653, 0.0047392
6: -0.0016294, 0.0000808, -0.0015920, 0.0000998, -0.0012029, 0.0011841
7: -0.0073535, -0.0029284, -0.0072565, -0.0028795, -0.0031122, 0.0030637
8: -0.0034313, -0.0011042, -0.0033803, -0.0010785, -0.0016366, 0.0016111
9: -0.0005835, 0.0021149, -0.0006133, 0.0020557, -0.0018682, 0.0018978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014510
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014893
time: 0.79 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9893148, 0.9933016, 0.9890686, 0.9933162, -0.0029703, 0.0031050
1: -0.0039264, -0.0029330, -0.0039878, -0.0029294, -0.0007401, 0.0007737
2: 0.0054895, 0.0107540, 0.0054701, 0.0110790, -0.0041002, 0.0039222
3: -0.0061679, -0.0037717, -0.0063158, -0.0037629, -0.0017852, 0.0018662
4: 0.0015904, 0.0026093, 0.0015866, 0.0026722, -0.0007936, 0.0007591
5: 0.0058638, 0.0124852, 0.0058394, 0.0128939, -0.0051570, 0.0049332
6: -0.0016280, 0.0000525, -0.0017318, 0.0000587, -0.0012521, 0.0013089
7: -0.0073499, -0.0030017, -0.0076183, -0.0029857, -0.0032395, 0.0033865
8: -0.0034294, -0.0011427, -0.0035705, -0.0011343, -0.0017036, 0.0017809
9: -0.0005388, 0.0021127, -0.0005486, 0.0022763, -0.0020651, 0.0019755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0014127
time: 0.83 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0014590
time: 0.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.07 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0013911
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0013911
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014591, upper bound: 0.0013375
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0013803
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0013938
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014199
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0013496
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0013915
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014503
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014503
IS_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014128, upper bound: 0.0014460
IS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0014460
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014510
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014510, upper bound: 0.0014893
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0014127
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -0.0014590, upper bound: 0.0014590

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9894857, 0.9932311, 0.9893966, 0.9934298, -0.0027427, 0.0026390
1: -0.0038838, -0.0029506, -0.0039060, -0.0029011, -0.0006834, 0.0006576
2: 0.0055825, 0.0105283, 0.0053202, 0.0106458, -0.0034848, 0.0036217
3: -0.0060651, -0.0038140, -0.0061186, -0.0036946, -0.0016485, 0.0015861
4: 0.0016084, 0.0025656, 0.0015576, 0.0025884, -0.0006745, 0.0007010
5: 0.0059807, 0.0122012, 0.0056509, 0.0123491, -0.0043830, 0.0045552
6: -0.0015560, 0.0000229, -0.0015935, 0.0001066, -0.0011562, 0.0011124
7: -0.0071634, -0.0030785, -0.0072605, -0.0028619, -0.0029913, 0.0028782
8: -0.0033313, -0.0011831, -0.0033824, -0.0010692, -0.0015731, 0.0015136
9: -0.0004920, 0.0019990, -0.0006241, 0.0020582, -0.0017551, 0.0018241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013205
time: 0.78 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013654
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9894857, 0.9932311, 0.9890653, 0.9933213, -0.0026691, 0.0030085
1: -0.0038838, -0.0029506, -0.0039886, -0.0029281, -0.0006651, 0.0007496
2: 0.0055825, 0.0105283, 0.0054635, 0.0110834, -0.0039727, 0.0035245
3: -0.0060651, -0.0038140, -0.0063178, -0.0037598, -0.0016042, 0.0018082
4: 0.0016084, 0.0025656, 0.0015853, 0.0026731, -0.0007689, 0.0006822
5: 0.0059807, 0.0122012, 0.0058310, 0.0128994, -0.0049966, 0.0044329
6: -0.0015560, 0.0000229, -0.0017332, 0.0000609, -0.0011251, 0.0012682
7: -0.0071634, -0.0030785, -0.0076219, -0.0029802, -0.0029110, 0.0032812
8: -0.0033313, -0.0011831, -0.0035724, -0.0011314, -0.0015309, 0.0017255
9: -0.0004920, 0.0019990, -0.0005519, 0.0022785, -0.0020008, 0.0017751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013205
time: 0.77 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013654
time: 1.09 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9891879, 0.9931140, 0.9894969, 0.9934219, -0.0030954, 0.0026723
1: -0.0039581, -0.0029798, -0.0038810, -0.0029030, -0.0007713, 0.0006659
2: 0.0057372, 0.0109216, 0.0053305, 0.0105135, -0.0035287, 0.0040874
3: -0.0062442, -0.0038844, -0.0060584, -0.0036993, -0.0018604, 0.0016061
4: 0.0016383, 0.0026417, 0.0015596, 0.0025628, -0.0006830, 0.0007911
5: 0.0061753, 0.0126960, 0.0056638, 0.0121827, -0.0044382, 0.0051409
6: -0.0016815, -0.0000265, -0.0015513, 0.0001033, -0.0013048, 0.0011265
7: -0.0074883, -0.0032063, -0.0071512, -0.0028704, -0.0033760, 0.0029145
8: -0.0035022, -0.0012503, -0.0033249, -0.0010736, -0.0017754, 0.0015327
9: -0.0004141, 0.0021971, -0.0006189, 0.0019915, -0.0017772, 0.0020586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014487, upper bound: 0.0013375
time: 0.78 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014487, upper bound: 0.0013375
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9891771, 0.9931147, 0.9894527, 0.9935246, -0.0031492, 0.0027182
1: -0.0039608, -0.0029796, -0.0038921, -0.0028775, -0.0007847, 0.0006773
2: 0.0057363, 0.0109359, 0.0051950, 0.0105720, -0.0035894, 0.0041585
3: -0.0062507, -0.0038840, -0.0060850, -0.0036377, -0.0018928, 0.0016337
4: 0.0016381, 0.0026445, 0.0015334, 0.0025741, -0.0006947, 0.0008049
5: 0.0061742, 0.0127139, 0.0054934, 0.0122562, -0.0045145, 0.0052303
6: -0.0016861, -0.0000263, -0.0015699, 0.0001465, -0.0013275, 0.0011458
7: -0.0075001, -0.0032056, -0.0071995, -0.0027585, -0.0034347, 0.0029646
8: -0.0035084, -0.0012499, -0.0033503, -0.0010148, -0.0018063, 0.0015591
9: -0.0004145, 0.0022043, -0.0006871, 0.0020210, -0.0018078, 0.0020944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013803
time: 0.81 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013803
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893892, 0.9932064, 0.9894006, 0.9933869, -0.0028017, 0.0027197
1: -0.0039079, -0.0029567, -0.0039050, -0.0029118, -0.0006981, 0.0006777
2: 0.0056151, 0.0106558, 0.0053768, 0.0106407, -0.0035913, 0.0036997
3: -0.0061232, -0.0038289, -0.0061163, -0.0037204, -0.0016839, 0.0016346
4: 0.0016147, 0.0025903, 0.0015686, 0.0025874, -0.0006951, 0.0007161
5: 0.0060218, 0.0123617, 0.0057220, 0.0123426, -0.0045169, 0.0046532
6: -0.0015967, 0.0000124, -0.0015919, 0.0000885, -0.0011810, 0.0011464
7: -0.0072687, -0.0031054, -0.0072562, -0.0029086, -0.0030557, 0.0029662
8: -0.0033867, -0.0011973, -0.0033801, -0.0010938, -0.0016070, 0.0015599
9: -0.0004756, 0.0020632, -0.0005956, 0.0020556, -0.0018088, 0.0018634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013802, upper bound: 0.0013688
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014228, upper bound: 0.0013688
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9890332, 0.9931207, 0.9894006, 0.9933869, -0.0032596, 0.0026184
1: -0.0039966, -0.0029781, -0.0039050, -0.0029118, -0.0008122, 0.0006524
2: 0.0057283, 0.0111259, 0.0053768, 0.0106407, -0.0034575, 0.0043043
3: -0.0063371, -0.0038804, -0.0061163, -0.0037204, -0.0019591, 0.0015737
4: 0.0016366, 0.0026813, 0.0015686, 0.0025874, -0.0006692, 0.0008331
5: 0.0061642, 0.0129529, 0.0057220, 0.0123426, -0.0043486, 0.0054137
6: -0.0017468, -0.0000237, -0.0015919, 0.0000885, -0.0013741, 0.0011037
7: -0.0076570, -0.0031989, -0.0072562, -0.0029086, -0.0035551, 0.0028557
8: -0.0035909, -0.0012464, -0.0033801, -0.0010938, -0.0018696, 0.0015018
9: -0.0004185, 0.0023000, -0.0005956, 0.0020556, -0.0017414, 0.0021679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013802, upper bound: 0.0013915
time: 1.05 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014228, upper bound: 0.0013915
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9894181, 0.9932045, 0.9891670, 0.9932812, -0.0028879, 0.0029696
1: -0.0039007, -0.0029572, -0.0039633, -0.0029381, -0.0007196, 0.0007400
2: 0.0056176, 0.0106175, 0.0055165, 0.0109492, -0.0039214, 0.0038135
3: -0.0061058, -0.0038300, -0.0062567, -0.0037840, -0.0017357, 0.0017848
4: 0.0016152, 0.0025829, 0.0015956, 0.0026471, -0.0007590, 0.0007381
5: 0.0060250, 0.0123135, 0.0058978, 0.0127306, -0.0049321, 0.0047964
6: -0.0015845, 0.0000116, -0.0016903, 0.0000439, -0.0012174, 0.0012518
7: -0.0072371, -0.0031075, -0.0075110, -0.0030240, -0.0031497, 0.0032388
8: -0.0033701, -0.0011984, -0.0035141, -0.0011545, -0.0016564, 0.0017033
9: -0.0004743, 0.0020439, -0.0005252, 0.0022109, -0.0019750, 0.0019207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013496
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013496
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9894079, 0.9932051, 0.9891235, 0.9933798, -0.0029255, 0.0030077
1: -0.0039032, -0.0029571, -0.0039741, -0.0029135, -0.0007290, 0.0007494
2: 0.0056169, 0.0106310, 0.0053862, 0.0110066, -0.0039716, 0.0038631
3: -0.0061119, -0.0038297, -0.0062829, -0.0037247, -0.0017583, 0.0018077
4: 0.0016150, 0.0025855, 0.0015704, 0.0026582, -0.0007687, 0.0007477
5: 0.0060240, 0.0123305, 0.0057339, 0.0128029, -0.0049952, 0.0048588
6: -0.0015888, 0.0000119, -0.0017087, 0.0000855, -0.0012332, 0.0012678
7: -0.0072483, -0.0031069, -0.0075585, -0.0029164, -0.0031907, 0.0032803
8: -0.0033759, -0.0011980, -0.0035391, -0.0010978, -0.0016780, 0.0017251
9: -0.0004747, 0.0020507, -0.0005909, 0.0022399, -0.0020003, 0.0019457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013915
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013915
time: 1.17 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9894058, 0.9933192, 0.9893965, 0.9934561, -0.0028031, 0.0026677
1: -0.0039037, -0.0029286, -0.0039061, -0.0028945, -0.0006985, 0.0006647
2: 0.0054663, 0.0106338, 0.0052854, 0.0106462, -0.0035227, 0.0037015
3: -0.0061131, -0.0037611, -0.0061188, -0.0036788, -0.0016847, 0.0016034
4: 0.0015859, 0.0025860, 0.0015509, 0.0025884, -0.0006818, 0.0007164
5: 0.0058346, 0.0123339, 0.0056071, 0.0123496, -0.0044306, 0.0046555
6: -0.0015896, 0.0000600, -0.0015936, 0.0001177, -0.0011816, 0.0011245
7: -0.0072505, -0.0029825, -0.0072608, -0.0028331, -0.0030572, 0.0029095
8: -0.0033771, -0.0011326, -0.0033825, -0.0010541, -0.0016077, 0.0015301
9: -0.0005505, 0.0020521, -0.0006416, 0.0020584, -0.0017742, 0.0018642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013809
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014278, upper bound: 0.0014224
time: 0.83 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9894058, 0.9933192, 0.9890651, 0.9933499, -0.0027283, 0.0030468
1: -0.0039037, -0.0029286, -0.0039886, -0.0029210, -0.0006798, 0.0007592
2: 0.0054663, 0.0106338, 0.0054258, 0.0110837, -0.0040233, 0.0036027
3: -0.0061131, -0.0037611, -0.0063179, -0.0037427, -0.0016398, 0.0018312
4: 0.0015859, 0.0025860, 0.0015780, 0.0026731, -0.0007787, 0.0006973
5: 0.0058346, 0.0123339, 0.0057837, 0.0128999, -0.0050602, 0.0045312
6: -0.0015896, 0.0000600, -0.0017333, 0.0000729, -0.0011501, 0.0012843
7: -0.0072505, -0.0029825, -0.0076222, -0.0029491, -0.0029756, 0.0033230
8: -0.0033771, -0.0011326, -0.0035726, -0.0011150, -0.0015648, 0.0017475
9: -0.0005505, 0.0020521, -0.0005709, 0.0022787, -0.0020263, 0.0018145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013808
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014278, upper bound: 0.0014224
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9891739, 0.9932092, 0.9894270, 0.9934536, -0.0030820, 0.0027664
1: -0.0039615, -0.0029560, -0.0038985, -0.0028951, -0.0007680, 0.0006893
2: 0.0056115, 0.0109400, 0.0052887, 0.0106058, -0.0036531, 0.0040697
3: -0.0062525, -0.0038272, -0.0061004, -0.0036803, -0.0018524, 0.0016627
4: 0.0016140, 0.0026453, 0.0015515, 0.0025806, -0.0007070, 0.0007877
5: 0.0060172, 0.0127190, 0.0056113, 0.0122988, -0.0045946, 0.0051187
6: -0.0016874, 0.0000136, -0.0015807, 0.0001166, -0.0012992, 0.0011662
7: -0.0075034, -0.0031024, -0.0072275, -0.0028359, -0.0033614, 0.0030172
8: -0.0035101, -0.0011957, -0.0033650, -0.0010555, -0.0017677, 0.0015867
9: -0.0004774, 0.0022063, -0.0006399, 0.0020380, -0.0018399, 0.0020497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014021, upper bound: 0.0014460
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014021, upper bound: 0.0014460
time: 0.91 seconds

## BFS IS instance: IS_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9891304, 0.9933133, 0.9894165, 0.9934542, -0.0031227, 0.0027921
1: -0.0039724, -0.0029301, -0.0039011, -0.0028950, -0.0007781, 0.0006957
2: 0.0054740, 0.0109975, 0.0052880, 0.0106197, -0.0036869, 0.0041235
3: -0.0062787, -0.0037646, -0.0061068, -0.0036800, -0.0018768, 0.0016781
4: 0.0015874, 0.0026564, 0.0015514, 0.0025833, -0.0007136, 0.0007981
5: 0.0058443, 0.0127914, 0.0056104, 0.0123163, -0.0046372, 0.0051862
6: -0.0017058, 0.0000575, -0.0015852, 0.0001169, -0.0013163, 0.0011770
7: -0.0075509, -0.0029889, -0.0072390, -0.0028353, -0.0034057, 0.0030452
8: -0.0035351, -0.0011360, -0.0033710, -0.0010552, -0.0017910, 0.0016014
9: -0.0005466, 0.0022353, -0.0006403, 0.0020450, -0.0018569, 0.0020768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014487, upper bound: 0.0014460
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2

### Relational analysis result of IS_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014487, upper bound: 0.0014460
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893146, 0.9932938, 0.9894004, 0.9934136, -0.0028464, 0.0027458
1: -0.0039265, -0.0029350, -0.0039051, -0.0029051, -0.0007092, 0.0006842
2: 0.0054998, 0.0107543, 0.0053416, 0.0106410, -0.0036258, 0.0037586
3: -0.0061680, -0.0037764, -0.0061164, -0.0037044, -0.0017107, 0.0016503
4: 0.0015924, 0.0026094, 0.0015617, 0.0025874, -0.0007018, 0.0007275
5: 0.0058767, 0.0124855, 0.0056778, 0.0123430, -0.0045603, 0.0047273
6: -0.0016281, 0.0000493, -0.0015920, 0.0000998, -0.0011998, 0.0011574
7: -0.0073501, -0.0030102, -0.0072565, -0.0028795, -0.0031044, 0.0029947
8: -0.0034295, -0.0011472, -0.0033803, -0.0010785, -0.0016325, 0.0015749
9: -0.0005336, 0.0021128, -0.0006133, 0.0020557, -0.0018261, 0.0018930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013802, upper bound: 0.0014277
time: 0.87 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014228, upper bound: 0.0014277
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9889472, 0.9932242, 0.9894004, 0.9934136, -0.0032941, 0.0026420
1: -0.0040180, -0.0029523, -0.0039051, -0.0029051, -0.0008208, 0.0006583
2: 0.0055917, 0.0112394, 0.0053416, 0.0106410, -0.0034887, 0.0043499
3: -0.0063888, -0.0038182, -0.0061164, -0.0037044, -0.0019799, 0.0015879
4: 0.0016102, 0.0027033, 0.0015617, 0.0025874, -0.0006752, 0.0008419
5: 0.0059923, 0.0130957, 0.0056778, 0.0123430, -0.0043878, 0.0054710
6: -0.0017830, 0.0000199, -0.0015920, 0.0000998, -0.0013886, 0.0011137
7: -0.0077507, -0.0030861, -0.0072565, -0.0028795, -0.0035927, 0.0028814
8: -0.0036402, -0.0011871, -0.0033803, -0.0010785, -0.0018894, 0.0015153
9: -0.0004874, 0.0023571, -0.0006133, 0.0020557, -0.0017571, 0.0021908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013802, upper bound: 0.0014590
time: 0.85 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014228, upper bound: 0.0014591
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9893442, 0.9932998, 0.9891669, 0.9933098, -0.0029312, 0.0030102
1: -0.0039191, -0.0029335, -0.0039633, -0.0029310, -0.0007304, 0.0007501
2: 0.0054918, 0.0107152, 0.0054786, 0.0109493, -0.0039749, 0.0038707
3: -0.0061502, -0.0037728, -0.0062568, -0.0037667, -0.0017618, 0.0018092
4: 0.0015908, 0.0026018, 0.0015883, 0.0026471, -0.0007693, 0.0007492
5: 0.0058667, 0.0124364, 0.0058501, 0.0127308, -0.0049994, 0.0048683
6: -0.0016157, 0.0000518, -0.0016904, 0.0000560, -0.0012356, 0.0012689
7: -0.0073178, -0.0030036, -0.0075111, -0.0029927, -0.0031969, 0.0032830
8: -0.0034125, -0.0011437, -0.0035142, -0.0011380, -0.0016812, 0.0017265
9: -0.0005377, 0.0020931, -0.0005443, 0.0022110, -0.0020020, 0.0019495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014128
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014128
time: 0.89 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9893340, 0.9933004, 0.9891233, 0.9934106, -0.0029815, 0.0030438
1: -0.0039216, -0.0029333, -0.0039741, -0.0029059, -0.0007429, 0.0007584
2: 0.0054910, 0.0107285, 0.0053455, 0.0110069, -0.0040193, 0.0039370
3: -0.0061563, -0.0037724, -0.0062830, -0.0037062, -0.0017920, 0.0018294
4: 0.0015907, 0.0026044, 0.0015625, 0.0026582, -0.0007779, 0.0007620
5: 0.0058656, 0.0124531, 0.0056827, 0.0128032, -0.0050552, 0.0049517
6: -0.0016199, 0.0000521, -0.0017088, 0.0000985, -0.0012568, 0.0012831
7: -0.0073288, -0.0030029, -0.0075587, -0.0028828, -0.0032517, 0.0033197
8: -0.0034183, -0.0011433, -0.0035392, -0.0010802, -0.0017101, 0.0017458
9: -0.0005381, 0.0020998, -0.0006113, 0.0022400, -0.0020243, 0.0019829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014590
time: 0.91 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014591
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.27 seconds
IS_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013205
IS_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013654
IS_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013205
IS_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013654
IS_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014487, upper bound: 0.0013375
IS_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014487, upper bound: 0.0013375
IS_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013803
IS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013803
IS_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0013802, upper bound: 0.0013688
IS_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014228, upper bound: 0.0013688
IS_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0013802, upper bound: 0.0013915
IS_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014228, upper bound: 0.0013915
IS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013496
IS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013496
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013915
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013915
IS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013809
IS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014278, upper bound: 0.0014224
IS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014277, upper bound: 0.0013808
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014278, upper bound: 0.0014224
IS_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014021, upper bound: 0.0014460
IS_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014021, upper bound: 0.0014460
IS_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014487, upper bound: 0.0014460
IS_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014487, upper bound: 0.0014460
IS_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0013802, upper bound: 0.0014277
IS_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014228, upper bound: 0.0014277
IS_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0013802, upper bound: 0.0014590
IS_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0014228, upper bound: 0.0014591
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014128
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014128
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014590
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014591

## BFS IS instance: IS_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9894964, 0.9934248, -0.0027080, 0.0025411
1: -0.0038766, -0.0029508, -0.0038812, -0.0029023, -0.0006748, 0.0006332
2: 0.0055839, 0.0104897, 0.0053267, 0.0105143, -0.0033556, 0.0035759
3: -0.0060476, -0.0038147, -0.0060588, -0.0036976, -0.0016276, 0.0015273
4: 0.0016086, 0.0025582, 0.0015589, 0.0025629, -0.0006495, 0.0006921
5: 0.0059825, 0.0121528, 0.0056591, 0.0121836, -0.0042204, 0.0044976
6: -0.0015437, 0.0000224, -0.0015515, 0.0001045, -0.0011415, 0.0010712
7: -0.0071316, -0.0030797, -0.0071518, -0.0028672, -0.0029535, 0.0027715
8: -0.0033146, -0.0011837, -0.0033252, -0.0010720, -0.0015532, 0.0014575
9: -0.0004913, 0.0019796, -0.0006208, 0.0019919, -0.0016900, 0.0018010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014311, upper bound: 0.0013279
time: 0.79 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014311, upper bound: 0.0013279
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9894516, 0.9935219, -0.0027445, 0.0025849
1: -0.0038792, -0.0029507, -0.0038923, -0.0028781, -0.0006839, 0.0006441
2: 0.0055833, 0.0105038, 0.0051986, 0.0105733, -0.0034133, 0.0036241
3: -0.0060540, -0.0038144, -0.0060856, -0.0036393, -0.0016495, 0.0015536
4: 0.0016085, 0.0025609, 0.0015341, 0.0025743, -0.0006606, 0.0007014
5: 0.0059818, 0.0121705, 0.0054979, 0.0122578, -0.0042931, 0.0045581
6: -0.0015482, 0.0000226, -0.0015703, 0.0001454, -0.0011569, 0.0010896
7: -0.0071432, -0.0030792, -0.0072006, -0.0027614, -0.0029932, 0.0028192
8: -0.0033207, -0.0011835, -0.0033509, -0.0010163, -0.0015741, 0.0014826
9: -0.0004916, 0.0019867, -0.0006854, 0.0020216, -0.0017191, 0.0018253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013950, upper bound: 0.0013731
time: 0.83 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013950, upper bound: 0.0013730
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9891636, 0.9933147, -0.0026340, 0.0029056
1: -0.0038766, -0.0029508, -0.0039641, -0.0029297, -0.0006563, 0.0007240
2: 0.0055839, 0.0104897, 0.0054721, 0.0109537, -0.0038368, 0.0034782
3: -0.0060476, -0.0038147, -0.0062588, -0.0037638, -0.0015831, 0.0017464
4: 0.0016086, 0.0025582, 0.0015870, 0.0026480, -0.0007426, 0.0006732
5: 0.0059825, 0.0121528, 0.0058419, 0.0127364, -0.0048257, 0.0043746
6: -0.0015437, 0.0000224, -0.0016918, 0.0000581, -0.0011103, 0.0012248
7: -0.0071316, -0.0030797, -0.0075148, -0.0029873, -0.0028728, 0.0031690
8: -0.0033146, -0.0011837, -0.0035161, -0.0011351, -0.0015108, 0.0016665
9: -0.0004913, 0.0019796, -0.0005476, 0.0022133, -0.0019324, 0.0017518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0013205
time: 0.78 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0013205
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9891197, 0.9934136, -0.0026875, 0.0029451
1: -0.0038792, -0.0029507, -0.0039750, -0.0029051, -0.0006696, 0.0007338
2: 0.0055833, 0.0105038, 0.0053417, 0.0110116, -0.0038890, 0.0035488
3: -0.0060540, -0.0038144, -0.0062851, -0.0037044, -0.0016153, 0.0017701
4: 0.0016085, 0.0025609, 0.0015618, 0.0026592, -0.0007527, 0.0006869
5: 0.0059818, 0.0121705, 0.0056779, 0.0128092, -0.0048914, 0.0044634
6: -0.0015482, 0.0000226, -0.0017103, 0.0000997, -0.0011329, 0.0012415
7: -0.0071432, -0.0030792, -0.0075626, -0.0028796, -0.0029311, 0.0032121
8: -0.0033207, -0.0011835, -0.0035413, -0.0010785, -0.0015414, 0.0016892
9: -0.0004916, 0.0019867, -0.0006133, 0.0022424, -0.0019587, 0.0017874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013653
time: 0.84 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013654
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9891879, 0.9931140, 0.9895050, 0.9933551, -0.0030062, 0.0026590
1: -0.0039581, -0.0029798, -0.0038790, -0.0029197, -0.0007491, 0.0006626
2: 0.0057372, 0.0109216, 0.0054187, 0.0105029, -0.0035112, 0.0039696
3: -0.0062442, -0.0038844, -0.0060536, -0.0037395, -0.0018068, 0.0015981
4: 0.0016383, 0.0026417, 0.0015767, 0.0025607, -0.0006796, 0.0007683
5: 0.0061753, 0.0126960, 0.0057748, 0.0121693, -0.0044162, 0.0049928
6: -0.0016815, -0.0000265, -0.0015479, 0.0000751, -0.0012672, 0.0011209
7: -0.0074883, -0.0032063, -0.0071424, -0.0029433, -0.0032787, 0.0029000
8: -0.0035022, -0.0012503, -0.0033203, -0.0011120, -0.0017242, 0.0015251
9: -0.0004141, 0.0021971, -0.0005745, 0.0019862, -0.0017684, 0.0019993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013375
time: 0.82 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013375
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9891879, 0.9931140, 0.9894124, 0.9933368, -0.0029980, 0.0027502
1: -0.0039581, -0.0029798, -0.0039021, -0.0029243, -0.0007470, 0.0006853
2: 0.0057372, 0.0109216, 0.0054430, 0.0106252, -0.0036316, 0.0039589
3: -0.0062442, -0.0038844, -0.0061092, -0.0037506, -0.0018019, 0.0016529
4: 0.0016383, 0.0026417, 0.0015814, 0.0025844, -0.0007029, 0.0007662
5: 0.0061753, 0.0126960, 0.0058054, 0.0123231, -0.0045676, 0.0049792
6: -0.0016815, -0.0000265, -0.0015869, 0.0000674, -0.0012638, 0.0011593
7: -0.0074883, -0.0032063, -0.0072434, -0.0029633, -0.0032698, 0.0029995
8: -0.0035022, -0.0012503, -0.0033734, -0.0011225, -0.0017195, 0.0015774
9: -0.0004141, 0.0021971, -0.0005622, 0.0020478, -0.0018291, 0.0019939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013375
time: 1.09 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013375
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9891771, 0.9931147, 0.9895313, 0.9933848, -0.0030157, 0.0026472
1: -0.0039608, -0.0029796, -0.0038725, -0.0029123, -0.0007514, 0.0006596
2: 0.0057363, 0.0109359, 0.0053795, 0.0104680, -0.0034956, 0.0039822
3: -0.0062507, -0.0038840, -0.0060377, -0.0037216, -0.0018125, 0.0015910
4: 0.0016381, 0.0026445, 0.0015691, 0.0025539, -0.0006766, 0.0007707
5: 0.0061742, 0.0127139, 0.0057254, 0.0121254, -0.0043965, 0.0050085
6: -0.0016861, -0.0000263, -0.0015367, 0.0000877, -0.0012712, 0.0011159
7: -0.0075001, -0.0032056, -0.0071136, -0.0029108, -0.0032890, 0.0028871
8: -0.0035084, -0.0012499, -0.0033051, -0.0010949, -0.0017297, 0.0015183
9: -0.0004145, 0.0022043, -0.0005942, 0.0019686, -0.0017606, 0.0020056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013803
time: 0.84 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013803
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9891771, 0.9931147, 0.9894537, 0.9934830, -0.0030804, 0.0026864
1: -0.0039608, -0.0029796, -0.0038918, -0.0028878, -0.0007676, 0.0006694
2: 0.0057363, 0.0109359, 0.0052498, 0.0105706, -0.0035473, 0.0040677
3: -0.0062507, -0.0038840, -0.0060844, -0.0036626, -0.0018514, 0.0016146
4: 0.0016381, 0.0026445, 0.0015440, 0.0025738, -0.0006866, 0.0007873
5: 0.0061742, 0.0127139, 0.0055624, 0.0122545, -0.0044616, 0.0051161
6: -0.0016861, -0.0000263, -0.0015695, 0.0001290, -0.0012985, 0.0011324
7: -0.0075001, -0.0032056, -0.0071984, -0.0028038, -0.0033597, 0.0029299
8: -0.0035084, -0.0012499, -0.0033497, -0.0010386, -0.0017668, 0.0015408
9: -0.0004145, 0.0022043, -0.0006595, 0.0020203, -0.0017866, 0.0020487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013803
time: 0.86 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013803
time: 0.88 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9894842, 0.9932004, 0.9894304, 0.9933856, -0.0026364, 0.0026815
1: -0.0038842, -0.0029582, -0.0038976, -0.0029121, -0.0006569, 0.0006682
2: 0.0056230, 0.0105302, 0.0053785, 0.0106014, -0.0035409, 0.0034814
3: -0.0060660, -0.0038325, -0.0060984, -0.0037212, -0.0015846, 0.0016116
4: 0.0016162, 0.0025660, 0.0015689, 0.0025798, -0.0006853, 0.0006738
5: 0.0060317, 0.0122036, 0.0057242, 0.0122932, -0.0044535, 0.0043786
6: -0.0015566, 0.0000099, -0.0015793, 0.0000880, -0.0011113, 0.0011303
7: -0.0071650, -0.0031119, -0.0072238, -0.0029100, -0.0028754, 0.0029245
8: -0.0033321, -0.0012007, -0.0033631, -0.0010945, -0.0015121, 0.0015380
9: -0.0004716, 0.0019999, -0.0005947, 0.0020358, -0.0017834, 0.0017534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013493, upper bound: 0.0013758
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013493, upper bound: 0.0013758
time: 1.10 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9894425, 0.9932938, 0.9894198, 0.9933861, -0.0027530, 0.0027185
1: -0.0038946, -0.0029350, -0.0039003, -0.0029120, -0.0006860, 0.0006774
2: 0.0054998, 0.0105854, 0.0053779, 0.0106154, -0.0035897, 0.0036353
3: -0.0060911, -0.0037764, -0.0061048, -0.0037209, -0.0016546, 0.0016339
4: 0.0015924, 0.0025767, 0.0015688, 0.0025825, -0.0006948, 0.0007036
5: 0.0058767, 0.0122731, 0.0057234, 0.0123108, -0.0045149, 0.0045723
6: -0.0015742, 0.0000493, -0.0015838, 0.0000882, -0.0011605, 0.0011459
7: -0.0072106, -0.0030102, -0.0072354, -0.0029095, -0.0030025, 0.0029649
8: -0.0033561, -0.0011472, -0.0033691, -0.0010942, -0.0015790, 0.0015592
9: -0.0005337, 0.0020277, -0.0005951, 0.0020428, -0.0018080, 0.0018309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013950, upper bound: 0.0013758
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013950, upper bound: 0.0013758
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9891263, 0.9931122, 0.9894304, 0.9933856, -0.0030996, 0.0025763
1: -0.0039734, -0.0029802, -0.0038976, -0.0029121, -0.0007723, 0.0006419
2: 0.0057396, 0.0110029, 0.0053785, 0.0106014, -0.0034019, 0.0040930
3: -0.0062812, -0.0038855, -0.0060984, -0.0037212, -0.0018630, 0.0015484
4: 0.0016388, 0.0026575, 0.0015689, 0.0025798, -0.0006584, 0.0007922
5: 0.0061784, 0.0127982, 0.0057242, 0.0122932, -0.0042787, 0.0051480
6: -0.0017075, -0.0000273, -0.0015793, 0.0000880, -0.0013066, 0.0010860
7: -0.0075554, -0.0032083, -0.0072238, -0.0029100, -0.0033806, 0.0028098
8: -0.0035375, -0.0012513, -0.0033631, -0.0010945, -0.0017778, 0.0014776
9: -0.0004129, 0.0022380, -0.0005947, 0.0020358, -0.0017134, 0.0020615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013405, upper bound: 0.0013915
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013405, upper bound: 0.0013915
time: 1.07 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9890873, 0.9932092, 0.9894198, 0.9933861, -0.0032048, 0.0026308
1: -0.0039831, -0.0029560, -0.0039003, -0.0029120, -0.0007986, 0.0006555
2: 0.0056114, 0.0110544, 0.0053779, 0.0106154, -0.0034740, 0.0042319
3: -0.0063046, -0.0038272, -0.0061048, -0.0037209, -0.0019262, 0.0015812
4: 0.0016140, 0.0026675, 0.0015688, 0.0025825, -0.0006724, 0.0008191
5: 0.0060171, 0.0128630, 0.0057234, 0.0123108, -0.0043693, 0.0053226
6: -0.0017239, 0.0000136, -0.0015838, 0.0000882, -0.0013509, 0.0011090
7: -0.0075980, -0.0031024, -0.0072354, -0.0029095, -0.0034953, 0.0028693
8: -0.0035599, -0.0011957, -0.0033691, -0.0010942, -0.0018381, 0.0015089
9: -0.0004774, 0.0022640, -0.0005951, 0.0020428, -0.0017497, 0.0021314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013854, upper bound: 0.0013915
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013854, upper bound: 0.0013915
time: 1.14 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9894181, 0.9932045, 0.9892498, 0.9931434, -0.0027626, 0.0029050
1: -0.0039007, -0.0029572, -0.0039426, -0.0029724, -0.0006884, 0.0007238
2: 0.0056176, 0.0106175, 0.0056983, 0.0108399, -0.0038360, 0.0036480
3: -0.0061058, -0.0038300, -0.0062070, -0.0038667, -0.0016604, 0.0017460
4: 0.0016152, 0.0025829, 0.0016308, 0.0026259, -0.0007425, 0.0007061
5: 0.0060250, 0.0123135, 0.0061264, 0.0125932, -0.0048247, 0.0045882
6: -0.0015845, 0.0000116, -0.0016554, -0.0000141, -0.0011645, 0.0012246
7: -0.0072371, -0.0031075, -0.0074208, -0.0031741, -0.0030130, 0.0031683
8: -0.0033701, -0.0011984, -0.0034667, -0.0012334, -0.0015845, 0.0016662
9: -0.0004743, 0.0020439, -0.0004337, 0.0021559, -0.0019320, 0.0018373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013677, upper bound: 0.0013189
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013144
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9894181, 0.9932045, 0.9891681, 0.9932417, -0.0028078, 0.0029358
1: -0.0039007, -0.0029572, -0.0039630, -0.0029480, -0.0006996, 0.0007315
2: 0.0056176, 0.0106175, 0.0055686, 0.0109477, -0.0038766, 0.0037077
3: -0.0061058, -0.0038300, -0.0062560, -0.0038077, -0.0016876, 0.0017645
4: 0.0016152, 0.0025829, 0.0016057, 0.0026468, -0.0007503, 0.0007176
5: 0.0060250, 0.0123135, 0.0059633, 0.0127287, -0.0048758, 0.0046633
6: -0.0015845, 0.0000116, -0.0016899, 0.0000273, -0.0011836, 0.0012375
7: -0.0072371, -0.0031075, -0.0075098, -0.0030670, -0.0030623, 0.0032019
8: -0.0033701, -0.0011984, -0.0035135, -0.0011771, -0.0016104, 0.0016838
9: -0.0004743, 0.0020439, -0.0004990, 0.0022102, -0.0019525, 0.0018674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013887, upper bound: 0.0013140
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013144
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9894079, 0.9932051, 0.9892061, 0.9932399, -0.0027958, 0.0029425
1: -0.0039032, -0.0029571, -0.0039535, -0.0029484, -0.0006967, 0.0007332
2: 0.0056169, 0.0106310, 0.0055709, 0.0108976, -0.0038856, 0.0036919
3: -0.0061119, -0.0038297, -0.0062332, -0.0038088, -0.0016804, 0.0017685
4: 0.0016150, 0.0025855, 0.0016061, 0.0026371, -0.0007520, 0.0007146
5: 0.0060240, 0.0123305, 0.0059662, 0.0126657, -0.0048870, 0.0046434
6: -0.0015888, 0.0000119, -0.0016739, 0.0000265, -0.0011785, 0.0012404
7: -0.0072483, -0.0031069, -0.0074684, -0.0030689, -0.0030493, 0.0032092
8: -0.0033759, -0.0011980, -0.0034917, -0.0011781, -0.0016036, 0.0016877
9: -0.0004747, 0.0020507, -0.0004978, 0.0021850, -0.0019570, 0.0018594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013676, upper bound: 0.0013606
time: 0.92 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013555
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9894079, 0.9932051, 0.9891247, 0.9933434, -0.0028632, 0.0029753
1: -0.0039032, -0.0029571, -0.0039738, -0.0029226, -0.0007134, 0.0007414
2: 0.0056169, 0.0106310, 0.0054343, 0.0110051, -0.0039289, 0.0037809
3: -0.0061119, -0.0038297, -0.0062821, -0.0037466, -0.0017209, 0.0017882
4: 0.0016150, 0.0025855, 0.0015797, 0.0026579, -0.0007604, 0.0007318
5: 0.0060240, 0.0123305, 0.0057944, 0.0128009, -0.0049415, 0.0047554
6: -0.0015888, 0.0000119, -0.0017082, 0.0000702, -0.0012070, 0.0012542
7: -0.0072483, -0.0031069, -0.0075572, -0.0029561, -0.0031228, 0.0032450
8: -0.0033759, -0.0011980, -0.0035384, -0.0011187, -0.0016422, 0.0017065
9: -0.0004747, 0.0020507, -0.0005666, 0.0022391, -0.0019788, 0.0019043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013887, upper bound: 0.0013536
time: 1.28 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013555
time: 0.85 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9894355, 0.9933180, 0.9894961, 0.9934511, -0.0027648, 0.0025802
1: -0.0038963, -0.0029289, -0.0038812, -0.0028958, -0.0006889, 0.0006429
2: 0.0054679, 0.0105945, 0.0052921, 0.0105145, -0.0034071, 0.0036508
3: -0.0060953, -0.0037619, -0.0060589, -0.0036819, -0.0016617, 0.0015508
4: 0.0015862, 0.0025784, 0.0015522, 0.0025629, -0.0006594, 0.0007066
5: 0.0058366, 0.0122845, 0.0056155, 0.0121839, -0.0042852, 0.0045918
6: -0.0015771, 0.0000594, -0.0015516, 0.0001156, -0.0011654, 0.0010876
7: -0.0072181, -0.0029838, -0.0071520, -0.0028386, -0.0030154, 0.0028140
8: -0.0033601, -0.0011333, -0.0033253, -0.0010570, -0.0015858, 0.0014799
9: -0.0005497, 0.0020323, -0.0006383, 0.0019920, -0.0017160, 0.0018388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014311, upper bound: 0.0013879
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014311, upper bound: 0.0013879
time: 1.00 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9894249, 0.9933185, 0.9894514, 0.9935510, -0.0028135, 0.0026209
1: -0.0038990, -0.0029288, -0.0038924, -0.0028709, -0.0007010, 0.0006530
2: 0.0054672, 0.0106086, 0.0051601, 0.0105736, -0.0034608, 0.0037152
3: -0.0061017, -0.0037616, -0.0060858, -0.0036218, -0.0016910, 0.0015752
4: 0.0015861, 0.0025812, 0.0015266, 0.0025744, -0.0006698, 0.0007191
5: 0.0058358, 0.0123023, 0.0054495, 0.0122583, -0.0043528, 0.0046727
6: -0.0015816, 0.0000596, -0.0015705, 0.0001577, -0.0011860, 0.0011048
7: -0.0072298, -0.0029833, -0.0072009, -0.0027296, -0.0030685, 0.0028584
8: -0.0033662, -0.0011330, -0.0033510, -0.0009996, -0.0016137, 0.0015032
9: -0.0005500, 0.0020394, -0.0007047, 0.0020218, -0.0017430, 0.0018712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013758, upper bound: 0.0014312
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013758, upper bound: 0.0014312
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9894355, 0.9933180, 0.9891635, 0.9933432, -0.0026892, 0.0029513
1: -0.0038963, -0.0029289, -0.0039641, -0.0029226, -0.0006701, 0.0007354
2: 0.0054679, 0.0105945, 0.0054345, 0.0109539, -0.0038971, 0.0035510
3: -0.0060953, -0.0037619, -0.0062589, -0.0037467, -0.0016163, 0.0017738
4: 0.0015862, 0.0025784, 0.0015797, 0.0026480, -0.0007543, 0.0006873
5: 0.0058366, 0.0122845, 0.0057946, 0.0127366, -0.0049016, 0.0044663
6: -0.0015771, 0.0000594, -0.0016919, 0.0000701, -0.0011336, 0.0012441
7: -0.0072181, -0.0029838, -0.0075150, -0.0029563, -0.0029329, 0.0032188
8: -0.0033601, -0.0011333, -0.0035162, -0.0011188, -0.0015424, 0.0016927
9: -0.0005497, 0.0020323, -0.0005665, 0.0022133, -0.0019628, 0.0017885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0013809
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0013808
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9894249, 0.9933185, 0.9891194, 0.9934449, -0.0027553, 0.0029866
1: -0.0038990, -0.0029288, -0.0039751, -0.0028973, -0.0006865, 0.0007442
2: 0.0054672, 0.0106086, 0.0053002, 0.0110119, -0.0039437, 0.0036383
3: -0.0061017, -0.0037616, -0.0062853, -0.0036855, -0.0016560, 0.0017950
4: 0.0015861, 0.0025812, 0.0015537, 0.0026592, -0.0007633, 0.0007042
5: 0.0058358, 0.0123023, 0.0056257, 0.0128096, -0.0049602, 0.0045761
6: -0.0015816, 0.0000596, -0.0017104, 0.0001130, -0.0011615, 0.0012589
7: -0.0072298, -0.0029833, -0.0075629, -0.0028453, -0.0030050, 0.0032573
8: -0.0033662, -0.0011330, -0.0035414, -0.0010605, -0.0015803, 0.0017130
9: -0.0005500, 0.0020394, -0.0006342, 0.0022426, -0.0019863, 0.0018325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014224
time: 0.87 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014224
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9891739, 0.9932092, 0.9894353, 0.9933864, -0.0029930, 0.0027534
1: -0.0039615, -0.0029560, -0.0038964, -0.0029119, -0.0007458, 0.0006861
2: 0.0056115, 0.0109400, 0.0053775, 0.0105949, -0.0036358, 0.0039522
3: -0.0062525, -0.0038272, -0.0060955, -0.0037207, -0.0017989, 0.0016549
4: 0.0016140, 0.0026453, 0.0015687, 0.0025785, -0.0007037, 0.0007649
5: 0.0060172, 0.0127190, 0.0057229, 0.0122851, -0.0045729, 0.0049708
6: -0.0016874, 0.0000136, -0.0015773, 0.0000883, -0.0012617, 0.0011606
7: -0.0075034, -0.0031024, -0.0072184, -0.0029092, -0.0032643, 0.0030029
8: -0.0035101, -0.0011957, -0.0033603, -0.0010940, -0.0017167, 0.0015792
9: -0.0004774, 0.0022063, -0.0005953, 0.0020325, -0.0018312, 0.0019905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013388, upper bound: 0.0014460
time: 1.18 seconds

## Relational analysis of IS_A2_A1_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013388, upper bound: 0.0014460
time: 0.84 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9891739, 0.9932092, 0.9893427, 0.9933693, -0.0029832, 0.0028616
1: -0.0039615, -0.0029560, -0.0039195, -0.0029161, -0.0007433, 0.0007130
2: 0.0056115, 0.0109400, 0.0054000, 0.0107171, -0.0037787, 0.0039393
3: -0.0062525, -0.0038272, -0.0061511, -0.0037310, -0.0017930, 0.0017199
4: 0.0016140, 0.0026453, 0.0015731, 0.0026022, -0.0007314, 0.0007624
5: 0.0060172, 0.0127190, 0.0057513, 0.0124387, -0.0047526, 0.0049546
6: -0.0016874, 0.0000136, -0.0016163, 0.0000811, -0.0012575, 0.0012063
7: -0.0075034, -0.0031024, -0.0073194, -0.0029278, -0.0032536, 0.0031210
8: -0.0035101, -0.0011957, -0.0034133, -0.0011038, -0.0017110, 0.0016413
9: -0.0004774, 0.0022063, -0.0005839, 0.0020941, -0.0019032, 0.0019840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013388, upper bound: 0.0014460
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013388, upper bound: 0.0014460
time: 0.85 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9891304, 0.9933133, 0.9894246, 0.9933870, -0.0030337, 0.0027790
1: -0.0039724, -0.0029301, -0.0038991, -0.0029118, -0.0007559, 0.0006924
2: 0.0054740, 0.0109975, 0.0053768, 0.0106090, -0.0036696, 0.0040060
3: -0.0062787, -0.0037646, -0.0061019, -0.0037204, -0.0018233, 0.0016702
4: 0.0015874, 0.0026564, 0.0015686, 0.0025812, -0.0007102, 0.0007753
5: 0.0058443, 0.0127914, 0.0057220, 0.0123028, -0.0046154, 0.0050385
6: -0.0017058, 0.0000575, -0.0015818, 0.0000885, -0.0012788, 0.0011714
7: -0.0075509, -0.0029889, -0.0072301, -0.0029086, -0.0033087, 0.0030309
8: -0.0035351, -0.0011360, -0.0033664, -0.0010937, -0.0017400, 0.0015939
9: -0.0005466, 0.0022353, -0.0005956, 0.0020396, -0.0018482, 0.0020176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A2_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014460
time: 0.81 seconds

## Relational analysis of IS_A2_A1_A2_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014460
time: 0.86 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9891304, 0.9933133, 0.9893327, 0.9933700, -0.0030241, 0.0028889
1: -0.0039724, -0.0029301, -0.0039220, -0.0029160, -0.0007535, 0.0007198
2: 0.0054740, 0.0109975, 0.0053991, 0.0107305, -0.0038148, 0.0039933
3: -0.0062787, -0.0037646, -0.0061572, -0.0037306, -0.0018176, 0.0017363
4: 0.0015874, 0.0026564, 0.0015729, 0.0026047, -0.0007383, 0.0007729
5: 0.0058443, 0.0127914, 0.0057501, 0.0124556, -0.0047980, 0.0050226
6: -0.0017058, 0.0000575, -0.0016205, 0.0000814, -0.0012748, 0.0012178
7: -0.0075509, -0.0029889, -0.0073304, -0.0029270, -0.0032982, 0.0031507
8: -0.0035351, -0.0011360, -0.0034191, -0.0011034, -0.0017345, 0.0016569
9: -0.0005466, 0.0022353, -0.0005843, 0.0021008, -0.0019213, 0.0020112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A2_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014460
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014460
time: 0.88 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.9894134, 0.9932876, 0.9894301, 0.9934123, -0.0027473, 0.0027103
1: -0.0039018, -0.0029365, -0.0038977, -0.0029055, -0.0006846, 0.0006753
2: 0.0055080, 0.0106237, 0.0053434, 0.0106017, -0.0035790, 0.0036278
3: -0.0061086, -0.0037801, -0.0060985, -0.0037052, -0.0016512, 0.0016290
4: 0.0015939, 0.0025841, 0.0015621, 0.0025798, -0.0006927, 0.0007022
5: 0.0058871, 0.0123213, 0.0056800, 0.0122935, -0.0045014, 0.0045629
6: -0.0015864, 0.0000466, -0.0015794, 0.0000992, -0.0011581, 0.0011425
7: -0.0072422, -0.0030170, -0.0072240, -0.0028810, -0.0029964, 0.0029560
8: -0.0033728, -0.0011507, -0.0033632, -0.0010792, -0.0015758, 0.0015545
9: -0.0005295, 0.0020470, -0.0006124, 0.0020359, -0.0018026, 0.0018272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013324, upper bound: 0.0014339
time: 0.85 seconds

## Relational analysis of IS_A2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013324, upper bound: 0.0014339
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.9893707, 0.9933856, 0.9894195, 0.9934127, -0.0027899, 0.0027394
1: -0.0039125, -0.0029121, -0.0039003, -0.0029053, -0.0006952, 0.0006826
2: 0.0053786, 0.0106801, 0.0053427, 0.0106157, -0.0036173, 0.0036840
3: -0.0061342, -0.0037212, -0.0061049, -0.0037049, -0.0016768, 0.0016464
4: 0.0015689, 0.0025950, 0.0015620, 0.0025825, -0.0007001, 0.0007130
5: 0.0057243, 0.0123922, 0.0056791, 0.0123112, -0.0045496, 0.0046335
6: -0.0016044, 0.0000880, -0.0015839, 0.0000994, -0.0011760, 0.0011547
7: -0.0072888, -0.0029101, -0.0072356, -0.0028804, -0.0030427, 0.0029877
8: -0.0033973, -0.0010945, -0.0033693, -0.0010789, -0.0016001, 0.0015712
9: -0.0005947, 0.0020754, -0.0006128, 0.0020430, -0.0018219, 0.0018554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013758, upper bound: 0.0014340
time: 0.84 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013758, upper bound: 0.0014340
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.9890425, 0.9932153, 0.9894301, 0.9934123, -0.0031946, 0.0026029
1: -0.0039943, -0.0029545, -0.0038977, -0.0029055, -0.0007960, 0.0006486
2: 0.0056035, 0.0111135, 0.0053434, 0.0106017, -0.0034371, 0.0042184
3: -0.0063315, -0.0038236, -0.0060985, -0.0037052, -0.0019200, 0.0015644
4: 0.0016124, 0.0026789, 0.0015621, 0.0025798, -0.0006652, 0.0008165
5: 0.0060072, 0.0129373, 0.0056800, 0.0122935, -0.0043229, 0.0053056
6: -0.0017428, 0.0000161, -0.0015794, 0.0000992, -0.0013466, 0.0010972
7: -0.0076468, -0.0030959, -0.0072240, -0.0028810, -0.0034841, 0.0028388
8: -0.0035855, -0.0011922, -0.0033632, -0.0010792, -0.0018323, 0.0014929
9: -0.0004814, 0.0022937, -0.0006124, 0.0020359, -0.0017311, 0.0021246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013240, upper bound: 0.0014590
time: 0.88 seconds

## Relational analysis of IS_A2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013240, upper bound: 0.0014590
time: 0.87 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.9890040, 0.9933147, 0.9894195, 0.9934127, -0.0032328, 0.0026496
1: -0.0040039, -0.0029298, -0.0039003, -0.0029053, -0.0008055, 0.0006602
2: 0.0054722, 0.0111644, 0.0053427, 0.0106157, -0.0034988, 0.0042689
3: -0.0063547, -0.0037638, -0.0061049, -0.0037049, -0.0019430, 0.0015925
4: 0.0015870, 0.0026887, 0.0015620, 0.0025825, -0.0006772, 0.0008262
5: 0.0058420, 0.0130013, 0.0056791, 0.0123112, -0.0044006, 0.0053692
6: -0.0017590, 0.0000581, -0.0015839, 0.0000994, -0.0013627, 0.0011169
7: -0.0076888, -0.0029874, -0.0072356, -0.0028804, -0.0035258, 0.0028898
8: -0.0036076, -0.0011352, -0.0033693, -0.0010789, -0.0018542, 0.0015197
9: -0.0005476, 0.0023194, -0.0006128, 0.0020430, -0.0017622, 0.0021500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013672, upper bound: 0.0014590
time: 0.84 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013672, upper bound: 0.0014591
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9893442, 0.9932998, 0.9892498, 0.9931434, -0.0027927, 0.0029599
1: -0.0039191, -0.0029335, -0.0039426, -0.0029724, -0.0006959, 0.0007375
2: 0.0054918, 0.0107152, 0.0056983, 0.0108399, -0.0039086, 0.0036878
3: -0.0061502, -0.0037728, -0.0062070, -0.0038667, -0.0016785, 0.0017790
4: 0.0015908, 0.0026018, 0.0016308, 0.0026259, -0.0007565, 0.0007138
5: 0.0058667, 0.0124364, 0.0061264, 0.0125932, -0.0049160, 0.0046383
6: -0.0016157, 0.0000518, -0.0016554, -0.0000141, -0.0011772, 0.0012477
7: -0.0073178, -0.0030036, -0.0074208, -0.0031741, -0.0030459, 0.0032282
8: -0.0034125, -0.0011437, -0.0034667, -0.0012334, -0.0016018, 0.0016977
9: -0.0005377, 0.0020931, -0.0004337, 0.0021559, -0.0019686, 0.0018574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0013840
time: 0.90 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0013720
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9893442, 0.9932998, 0.9891681, 0.9932417, -0.0028591, 0.0029909
1: -0.0039191, -0.0029335, -0.0039630, -0.0029480, -0.0007124, 0.0007452
2: 0.0054918, 0.0107152, 0.0055686, 0.0109477, -0.0039494, 0.0037754
3: -0.0061502, -0.0037728, -0.0062560, -0.0038077, -0.0017184, 0.0017976
4: 0.0015908, 0.0026018, 0.0016057, 0.0026468, -0.0007644, 0.0007307
5: 0.0058667, 0.0124364, 0.0059633, 0.0127287, -0.0049673, 0.0047485
6: -0.0016157, 0.0000518, -0.0016899, 0.0000273, -0.0012052, 0.0012608
7: -0.0073178, -0.0030036, -0.0075098, -0.0030670, -0.0031182, 0.0032620
8: -0.0034125, -0.0011437, -0.0035135, -0.0011771, -0.0016399, 0.0017154
9: -0.0005377, 0.0020931, -0.0004990, 0.0022102, -0.0019891, 0.0019015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0013840
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0013720
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9893340, 0.9933004, 0.9892061, 0.9932399, -0.0028298, 0.0029975
1: -0.0039216, -0.0029333, -0.0039535, -0.0029484, -0.0007051, 0.0007469
2: 0.0054910, 0.0107285, 0.0055709, 0.0108976, -0.0039582, 0.0037367
3: -0.0061563, -0.0037724, -0.0062332, -0.0038088, -0.0017008, 0.0018016
4: 0.0015907, 0.0026044, 0.0016061, 0.0026371, -0.0007661, 0.0007232
5: 0.0058656, 0.0124531, 0.0059662, 0.0126657, -0.0049783, 0.0046998
6: -0.0016199, 0.0000521, -0.0016739, 0.0000265, -0.0011929, 0.0012636
7: -0.0073288, -0.0030029, -0.0074684, -0.0030689, -0.0030863, 0.0032692
8: -0.0034183, -0.0011433, -0.0034917, -0.0011781, -0.0016231, 0.0017192
9: -0.0005381, 0.0020998, -0.0004978, 0.0021850, -0.0019935, 0.0018820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014299
time: 0.99 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014185
time: 0.88 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9893340, 0.9933004, 0.9891247, 0.9933434, -0.0028884, 0.0030248
1: -0.0039216, -0.0029333, -0.0039738, -0.0029226, -0.0007197, 0.0007537
2: 0.0054910, 0.0107285, 0.0054343, 0.0110051, -0.0039942, 0.0038141
3: -0.0061563, -0.0037724, -0.0062821, -0.0037466, -0.0017360, 0.0018180
4: 0.0015907, 0.0026044, 0.0015797, 0.0026579, -0.0007731, 0.0007382
5: 0.0058656, 0.0124531, 0.0057944, 0.0128009, -0.0050236, 0.0047972
6: -0.0016199, 0.0000521, -0.0017082, 0.0000702, -0.0012176, 0.0012751
7: -0.0073288, -0.0030029, -0.0075572, -0.0029561, -0.0031502, 0.0032989
8: -0.0034183, -0.0011433, -0.0035384, -0.0011187, -0.0016567, 0.0017349
9: -0.0005381, 0.0020998, -0.0005666, 0.0022391, -0.0020117, 0.0019210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014299
time: 0.87 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014185
time: 0.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.28 seconds
IS_A1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014311, upper bound: 0.0013279
IS_A1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014311, upper bound: 0.0013279
IS_A1_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013950, upper bound: 0.0013731
IS_A1_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013950, upper bound: 0.0013730
IS_A1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0013205
IS_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0013205
IS_A1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013653
IS_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014163, upper bound: 0.0013654
IS_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013375
IS_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013375
IS_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013375
IS_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013375
IS_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013803
IS_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013803
IS_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013803
IS_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013803
IS_A1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013493, upper bound: 0.0013758
IS_A1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013493, upper bound: 0.0013758
IS_A1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013950, upper bound: 0.0013758
IS_A1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013950, upper bound: 0.0013758
IS_A1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013405, upper bound: 0.0013915
IS_A1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013405, upper bound: 0.0013915
IS_A1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013854, upper bound: 0.0013915
IS_A1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013854, upper bound: 0.0013915
IS_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013677, upper bound: 0.0013189
IS_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013144
IS_A1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013887, upper bound: 0.0013140
IS_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013144
IS_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013676, upper bound: 0.0013606
IS_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013555
IS_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013887, upper bound: 0.0013536
IS_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013555
IS_A2_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014311, upper bound: 0.0013879
IS_A2_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0014311, upper bound: 0.0013879
IS_A2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013758, upper bound: 0.0014312
IS_A2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013758, upper bound: 0.0014312
IS_A2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0013809
IS_A2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0013808
IS_A2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014224
IS_A2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013915, upper bound: 0.0014224
IS_A2_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013388, upper bound: 0.0014460
IS_A2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013388, upper bound: 0.0014460
IS_A2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013388, upper bound: 0.0014460
IS_A2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013388, upper bound: 0.0014460
IS_A2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014460
IS_A2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014460
IS_A2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014460
IS_A2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014460
IS_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013324, upper bound: 0.0014339
IS_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013324, upper bound: 0.0014339
IS_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013758, upper bound: 0.0014340
IS_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013758, upper bound: 0.0014340
IS_A2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013240, upper bound: 0.0014590
IS_A2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013240, upper bound: 0.0014590
IS_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013672, upper bound: 0.0014590
IS_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013672, upper bound: 0.0014591
IS_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0013840
IS_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0013720
IS_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0013840
IS_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0013720
IS_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014299
IS_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014185
IS_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014299
IS_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014185

## BFS IS instance: IS_A1_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9895042, 0.9933605, -0.0026348, 0.0025282
1: -0.0038766, -0.0029508, -0.0038793, -0.0029183, -0.0006565, 0.0006300
2: 0.0055839, 0.0104897, 0.0054117, 0.0105040, -0.0033384, 0.0034793
3: -0.0060476, -0.0038147, -0.0060541, -0.0037363, -0.0015836, 0.0015195
4: 0.0016086, 0.0025582, 0.0015753, 0.0025609, -0.0006461, 0.0006734
5: 0.0059825, 0.0121528, 0.0057660, 0.0121707, -0.0041989, 0.0043760
6: -0.0015437, 0.0000224, -0.0015482, 0.0000774, -0.0011107, 0.0010657
7: -0.0071316, -0.0030797, -0.0071434, -0.0029374, -0.0028736, 0.0027573
8: -0.0033146, -0.0011837, -0.0033208, -0.0011089, -0.0015112, 0.0014501
9: -0.0004913, 0.0019796, -0.0005780, 0.0019867, -0.0016814, 0.0017523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013279
time: 0.83 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013279
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9894123, 0.9933344, -0.0025845, 0.0026235
1: -0.0038766, -0.0029508, -0.0039021, -0.0029249, -0.0006440, 0.0006537
2: 0.0055839, 0.0104897, 0.0054462, 0.0106253, -0.0034643, 0.0034128
3: -0.0060476, -0.0038147, -0.0061093, -0.0037520, -0.0015534, 0.0015768
4: 0.0016086, 0.0025582, 0.0015820, 0.0025844, -0.0006705, 0.0006605
5: 0.0059825, 0.0121528, 0.0058094, 0.0123232, -0.0043572, 0.0042925
6: -0.0015437, 0.0000224, -0.0015869, 0.0000664, -0.0010895, 0.0011059
7: -0.0071316, -0.0030797, -0.0072435, -0.0029659, -0.0028188, 0.0028613
8: -0.0033146, -0.0011837, -0.0033734, -0.0011239, -0.0014824, 0.0015047
9: -0.0004913, 0.0019796, -0.0005606, 0.0020478, -0.0017448, 0.0017189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013279
time: 0.78 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013279
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9895304, 0.9933859, -0.0026133, 0.0025135
1: -0.0038792, -0.0029507, -0.0038727, -0.0029120, -0.0006512, 0.0006263
2: 0.0055833, 0.0105038, 0.0053781, 0.0104693, -0.0033190, 0.0034508
3: -0.0060540, -0.0038144, -0.0060383, -0.0037210, -0.0015707, 0.0015107
4: 0.0016085, 0.0025609, 0.0015688, 0.0025542, -0.0006424, 0.0006679
5: 0.0059818, 0.0121705, 0.0057237, 0.0121270, -0.0041745, 0.0043403
6: -0.0015482, 0.0000226, -0.0015371, 0.0000881, -0.0011016, 0.0010595
7: -0.0071432, -0.0030792, -0.0071147, -0.0029097, -0.0028502, 0.0027413
8: -0.0033207, -0.0011835, -0.0033057, -0.0010943, -0.0014989, 0.0014416
9: -0.0004916, 0.0019867, -0.0005949, 0.0019693, -0.0016716, 0.0017380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013730
time: 0.78 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013731
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9894527, 0.9934784, -0.0026790, 0.0025531
1: -0.0038792, -0.0029507, -0.0038921, -0.0028889, -0.0006675, 0.0006362
2: 0.0055833, 0.0105038, 0.0052559, 0.0105719, -0.0033713, 0.0035376
3: -0.0060540, -0.0038144, -0.0060850, -0.0036654, -0.0016102, 0.0015345
4: 0.0016085, 0.0025609, 0.0015452, 0.0025741, -0.0006525, 0.0006847
5: 0.0059818, 0.0121705, 0.0055700, 0.0122561, -0.0042402, 0.0044494
6: -0.0015482, 0.0000226, -0.0015699, 0.0001271, -0.0011293, 0.0010762
7: -0.0071432, -0.0030792, -0.0071995, -0.0028088, -0.0029218, 0.0027845
8: -0.0033207, -0.0011835, -0.0033503, -0.0010413, -0.0015366, 0.0014643
9: -0.0004916, 0.0019867, -0.0006565, 0.0020209, -0.0016980, 0.0017817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013731
time: 0.78 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013731
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9892465, 0.9931781, -0.0025107, 0.0028406
1: -0.0038766, -0.0029508, -0.0039434, -0.0029638, -0.0006256, 0.0007078
2: 0.0055839, 0.0104897, 0.0056525, 0.0108441, -0.0037510, 0.0033153
3: -0.0060476, -0.0038147, -0.0062089, -0.0038459, -0.0015090, 0.0017073
4: 0.0016086, 0.0025582, 0.0016219, 0.0026267, -0.0007260, 0.0006417
5: 0.0059825, 0.0121528, 0.0060688, 0.0125985, -0.0047178, 0.0041698
6: -0.0015437, 0.0000224, -0.0016568, 0.0000005, -0.0010583, 0.0011974
7: -0.0071316, -0.0030797, -0.0074243, -0.0031363, -0.0027382, 0.0030981
8: -0.0033146, -0.0011837, -0.0034685, -0.0012135, -0.0014400, 0.0016293
9: -0.0004913, 0.0019796, -0.0004567, 0.0021580, -0.0018892, 0.0016698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013205
time: 0.80 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013205
time: 0.80 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9891647, 0.9932756, -0.0025546, 0.0028720
1: -0.0038766, -0.0029508, -0.0039638, -0.0029395, -0.0006365, 0.0007156
2: 0.0055839, 0.0104897, 0.0055239, 0.0109522, -0.0037925, 0.0033733
3: -0.0060476, -0.0038147, -0.0062581, -0.0037874, -0.0015354, 0.0017262
4: 0.0016086, 0.0025582, 0.0015970, 0.0026477, -0.0007340, 0.0006529
5: 0.0059825, 0.0121528, 0.0059071, 0.0127345, -0.0047700, 0.0042427
6: -0.0015437, 0.0000224, -0.0016913, 0.0000415, -0.0010768, 0.0012107
7: -0.0071316, -0.0030797, -0.0075136, -0.0030301, -0.0027861, 0.0031324
8: -0.0033146, -0.0011837, -0.0035155, -0.0011577, -0.0014652, 0.0016473
9: -0.0004913, 0.0019796, -0.0005215, 0.0022125, -0.0019101, 0.0016990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013205
time: 1.04 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013205
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9892024, 0.9932750, -0.0025607, 0.0028798
1: -0.0038792, -0.0029507, -0.0039544, -0.0029397, -0.0006381, 0.0007176
2: 0.0055833, 0.0105038, 0.0055247, 0.0109023, -0.0038028, 0.0033814
3: -0.0060540, -0.0038144, -0.0062354, -0.0037877, -0.0015391, 0.0017309
4: 0.0016085, 0.0025609, 0.0015972, 0.0026380, -0.0007360, 0.0006545
5: 0.0059818, 0.0121705, 0.0059080, 0.0126717, -0.0047829, 0.0042529
6: -0.0015482, 0.0000226, -0.0016754, 0.0000413, -0.0010794, 0.0012140
7: -0.0071432, -0.0030792, -0.0074723, -0.0030307, -0.0027928, 0.0031409
8: -0.0033207, -0.0011835, -0.0034938, -0.0011580, -0.0014687, 0.0016517
9: -0.0004916, 0.0019867, -0.0005211, 0.0021873, -0.0019153, 0.0017030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013654
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013654
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9891208, 0.9933779, -0.0026188, 0.0029131
1: -0.0038792, -0.0029507, -0.0039747, -0.0029140, -0.0006525, 0.0007259
2: 0.0055833, 0.0105038, 0.0053887, 0.0110101, -0.0038467, 0.0034581
3: -0.0060540, -0.0038144, -0.0062844, -0.0037258, -0.0015740, 0.0017509
4: 0.0016085, 0.0025609, 0.0015709, 0.0026589, -0.0007445, 0.0006693
5: 0.0059818, 0.0121705, 0.0057370, 0.0128072, -0.0048381, 0.0043494
6: -0.0015482, 0.0000226, -0.0017098, 0.0000847, -0.0011039, 0.0012280
7: -0.0071432, -0.0030792, -0.0075614, -0.0029184, -0.0028562, 0.0031771
8: -0.0033207, -0.0011835, -0.0035406, -0.0010989, -0.0015020, 0.0016708
9: -0.0004916, 0.0019867, -0.0005896, 0.0022416, -0.0019374, 0.0017417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013654
time: 0.86 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013654
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9891879, 0.9931140, 0.9895828, 0.9932191, -0.0028801, 0.0025885
1: -0.0039581, -0.0029798, -0.0038596, -0.0029536, -0.0007176, 0.0006450
2: 0.0057372, 0.0109216, 0.0055984, 0.0104001, -0.0034181, 0.0038031
3: -0.0062442, -0.0038844, -0.0060068, -0.0038213, -0.0017310, 0.0015558
4: 0.0016383, 0.0026417, 0.0016115, 0.0025408, -0.0006616, 0.0007361
5: 0.0061753, 0.0126960, 0.0060008, 0.0120400, -0.0042990, 0.0047833
6: -0.0016815, -0.0000265, -0.0015150, 0.0000178, -0.0012141, 0.0010911
7: -0.0074883, -0.0032063, -0.0070575, -0.0030917, -0.0031411, 0.0028231
8: -0.0035022, -0.0012503, -0.0032756, -0.0011900, -0.0016519, 0.0014847
9: -0.0004141, 0.0021971, -0.0004840, 0.0019344, -0.0017215, 0.0019155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013597, upper bound: 0.0013079
time: 0.82 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013682, upper bound: 0.0013051
time: 0.80 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9891879, 0.9931140, 0.9895060, 0.9933121, -0.0029269, 0.0026248
1: -0.0039581, -0.0029798, -0.0038788, -0.0029304, -0.0007293, 0.0006540
2: 0.0057372, 0.0109216, 0.0054755, 0.0105016, -0.0034660, 0.0038649
3: -0.0062442, -0.0038844, -0.0060530, -0.0037653, -0.0017591, 0.0015776
4: 0.0016383, 0.0026417, 0.0015877, 0.0025604, -0.0006708, 0.0007480
5: 0.0061753, 0.0126960, 0.0058462, 0.0121677, -0.0043593, 0.0048611
6: -0.0016815, -0.0000265, -0.0015475, 0.0000570, -0.0012338, 0.0011064
7: -0.0074883, -0.0032063, -0.0071414, -0.0029901, -0.0031922, 0.0028627
8: -0.0035022, -0.0012503, -0.0033197, -0.0011366, -0.0016787, 0.0015055
9: -0.0004141, 0.0021971, -0.0005459, 0.0019855, -0.0017457, 0.0019466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013597, upper bound: 0.0013079
time: 0.80 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013682, upper bound: 0.0013051
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9891879, 0.9931140, 0.9894842, 0.9932002, -0.0028753, 0.0026453
1: -0.0039581, -0.0029798, -0.0038842, -0.0029583, -0.0007165, 0.0006591
2: 0.0057372, 0.0109216, 0.0056233, 0.0105302, -0.0034931, 0.0037969
3: -0.0062442, -0.0038844, -0.0060660, -0.0038326, -0.0017282, 0.0015899
4: 0.0016383, 0.0026417, 0.0016163, 0.0025660, -0.0006761, 0.0007349
5: 0.0061753, 0.0126960, 0.0060320, 0.0122037, -0.0043933, 0.0047754
6: -0.0016815, -0.0000265, -0.0015566, 0.0000098, -0.0012121, 0.0011151
7: -0.0074883, -0.0032063, -0.0071650, -0.0031122, -0.0031360, 0.0028850
8: -0.0035022, -0.0012503, -0.0033322, -0.0012008, -0.0016492, 0.0015172
9: -0.0004141, 0.0021971, -0.0004715, 0.0019999, -0.0017593, 0.0019123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013676, upper bound: 0.0013056
time: 0.87 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013028
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9891879, 0.9931140, 0.9894134, 0.9932954, -0.0029162, 0.0027175
1: -0.0039581, -0.0029798, -0.0039018, -0.0029346, -0.0007266, 0.0006771
2: 0.0057372, 0.0109216, 0.0054977, 0.0106237, -0.0035885, 0.0038508
3: -0.0062442, -0.0038844, -0.0061086, -0.0037755, -0.0017527, 0.0016333
4: 0.0016383, 0.0026417, 0.0015920, 0.0025841, -0.0006945, 0.0007453
5: 0.0061753, 0.0126960, 0.0058742, 0.0123212, -0.0045134, 0.0048432
6: -0.0016815, -0.0000265, -0.0015864, 0.0000499, -0.0012293, 0.0011455
7: -0.0074883, -0.0032063, -0.0072422, -0.0030085, -0.0031805, 0.0029639
8: -0.0035022, -0.0012503, -0.0033728, -0.0011463, -0.0016726, 0.0015587
9: -0.0004141, 0.0021971, -0.0005347, 0.0020470, -0.0018073, 0.0019394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013676, upper bound: 0.0013056
time: 0.82 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013028
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9891771, 0.9931147, 0.9895397, 0.9933193, -0.0029241, 0.0026346
1: -0.0039608, -0.0029796, -0.0038704, -0.0029286, -0.0007286, 0.0006565
2: 0.0057363, 0.0109359, 0.0054661, 0.0104569, -0.0034790, 0.0038612
3: -0.0062507, -0.0038840, -0.0060327, -0.0037610, -0.0017574, 0.0015835
4: 0.0016381, 0.0026445, 0.0015858, 0.0025518, -0.0006734, 0.0007473
5: 0.0061742, 0.0127139, 0.0058343, 0.0121115, -0.0043757, 0.0048564
6: -0.0016861, -0.0000263, -0.0015332, 0.0000600, -0.0012326, 0.0011106
7: -0.0075001, -0.0032056, -0.0071045, -0.0029823, -0.0031891, 0.0028735
8: -0.0035084, -0.0012499, -0.0033003, -0.0011325, -0.0016771, 0.0015111
9: -0.0004145, 0.0022043, -0.0005506, 0.0019630, -0.0017522, 0.0019447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013594, upper bound: 0.0013766
time: 0.79 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013661
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9891771, 0.9931147, 0.9894430, 0.9932967, -0.0029228, 0.0027383
1: -0.0039608, -0.0029796, -0.0038945, -0.0029343, -0.0007283, 0.0006823
2: 0.0057363, 0.0109359, 0.0054960, 0.0105847, -0.0036159, 0.0038595
3: -0.0062507, -0.0038840, -0.0060908, -0.0037747, -0.0017567, 0.0016458
4: 0.0016381, 0.0026445, 0.0015916, 0.0025765, -0.0006998, 0.0007470
5: 0.0061742, 0.0127139, 0.0058720, 0.0122722, -0.0045478, 0.0048543
6: -0.0016861, -0.0000263, -0.0015740, 0.0000505, -0.0012321, 0.0011543
7: -0.0075001, -0.0032056, -0.0072100, -0.0030071, -0.0031877, 0.0029865
8: -0.0035084, -0.0012499, -0.0033558, -0.0011455, -0.0016764, 0.0015706
9: -0.0004145, 0.0022043, -0.0005355, 0.0020274, -0.0018212, 0.0019439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013594, upper bound: 0.0013766
time: 0.85 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013661
time: 1.09 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9891771, 0.9931147, 0.9894618, 0.9934186, -0.0029927, 0.0026726
1: -0.0039608, -0.0029796, -0.0038898, -0.0029039, -0.0007457, 0.0006659
2: 0.0057363, 0.0109359, 0.0053350, 0.0105598, -0.0035291, 0.0039518
3: -0.0062507, -0.0038840, -0.0060795, -0.0037014, -0.0017987, 0.0016063
4: 0.0016381, 0.0026445, 0.0015605, 0.0025717, -0.0006831, 0.0007649
5: 0.0061742, 0.0127139, 0.0056694, 0.0122409, -0.0044387, 0.0049703
6: -0.0016861, -0.0000263, -0.0015660, 0.0001019, -0.0012615, 0.0011266
7: -0.0075001, -0.0032056, -0.0071894, -0.0028741, -0.0032639, 0.0029149
8: -0.0035084, -0.0012499, -0.0033450, -0.0010756, -0.0017165, 0.0015329
9: -0.0004145, 0.0022043, -0.0006167, 0.0020148, -0.0017775, 0.0019903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0013444
time: 0.87 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0013448
time: 0.88 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9891771, 0.9931147, 0.9893712, 0.9933965, -0.0029770, 0.0027624
1: -0.0039608, -0.0029796, -0.0039124, -0.0029094, -0.0007418, 0.0006883
2: 0.0057363, 0.0109359, 0.0053642, 0.0106795, -0.0036477, 0.0039311
3: -0.0062507, -0.0038840, -0.0061340, -0.0037147, -0.0017893, 0.0016603
4: 0.0016381, 0.0026445, 0.0015661, 0.0025949, -0.0007060, 0.0007609
5: 0.0061742, 0.0127139, 0.0057062, 0.0123914, -0.0045878, 0.0049443
6: -0.0016861, -0.0000263, -0.0016042, 0.0000925, -0.0012549, 0.0011644
7: -0.0075001, -0.0032056, -0.0072883, -0.0028982, -0.0032468, 0.0030127
8: -0.0035084, -0.0012499, -0.0033970, -0.0010883, -0.0017075, 0.0015844
9: -0.0004145, 0.0022043, -0.0006019, 0.0020751, -0.0018372, 0.0019799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0013444
time: 1.00 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0013448
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9894842, 0.9932004, 0.9895107, 0.9932514, -0.0025210, 0.0026041
1: -0.0038842, -0.0029582, -0.0038776, -0.0029455, -0.0006282, 0.0006489
2: 0.0056230, 0.0105302, 0.0055557, 0.0104952, -0.0034387, 0.0033289
3: -0.0060660, -0.0038325, -0.0060501, -0.0038018, -0.0015152, 0.0015651
4: 0.0016162, 0.0025660, 0.0016032, 0.0025592, -0.0006655, 0.0006443
5: 0.0060317, 0.0122036, 0.0059470, 0.0121597, -0.0043249, 0.0041869
6: -0.0015566, 0.0000099, -0.0015454, 0.0000314, -0.0010627, 0.0010977
7: -0.0071650, -0.0031119, -0.0071361, -0.0030563, -0.0027495, 0.0028401
8: -0.0033321, -0.0012007, -0.0033170, -0.0011714, -0.0014459, 0.0014936
9: -0.0004716, 0.0019999, -0.0005055, 0.0019823, -0.0017319, 0.0016766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013211, upper bound: 0.0013386
time: 0.83 seconds

## Relational analysis of IS_A1_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013120, upper bound: 0.0013404
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9894842, 0.9932004, 0.9894315, 0.9933367, -0.0025710, 0.0026472
1: -0.0038842, -0.0029582, -0.0038973, -0.0029243, -0.0006406, 0.0006596
2: 0.0056230, 0.0105302, 0.0054432, 0.0105998, -0.0034956, 0.0033950
3: -0.0060660, -0.0038325, -0.0060977, -0.0037506, -0.0015452, 0.0015910
4: 0.0016162, 0.0025660, 0.0015814, 0.0025795, -0.0006766, 0.0006571
5: 0.0060317, 0.0122036, 0.0058055, 0.0122913, -0.0043966, 0.0042700
6: -0.0015566, 0.0000099, -0.0015788, 0.0000673, -0.0010838, 0.0011159
7: -0.0071650, -0.0031119, -0.0072225, -0.0029634, -0.0028040, 0.0028872
8: -0.0033321, -0.0012007, -0.0033624, -0.0011226, -0.0014746, 0.0015183
9: -0.0004716, 0.0019999, -0.0005622, 0.0020350, -0.0017606, 0.0017099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013211, upper bound: 0.0013386
time: 0.83 seconds

## Relational analysis of IS_A1_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013120, upper bound: 0.0013403
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9894425, 0.9932938, 0.9895000, 0.9932519, -0.0026289, 0.0026343
1: -0.0038946, -0.0029350, -0.0038803, -0.0029454, -0.0006551, 0.0006564
2: 0.0054998, 0.0105854, 0.0055551, 0.0105094, -0.0034786, 0.0034715
3: -0.0060911, -0.0037764, -0.0060565, -0.0038016, -0.0015801, 0.0015833
4: 0.0015924, 0.0025767, 0.0016031, 0.0025620, -0.0006733, 0.0006719
5: 0.0058767, 0.0122731, 0.0059463, 0.0121775, -0.0043751, 0.0043662
6: -0.0015742, 0.0000493, -0.0015499, 0.0000316, -0.0011082, 0.0011105
7: -0.0072106, -0.0030102, -0.0071478, -0.0030559, -0.0028672, 0.0028731
8: -0.0033561, -0.0011472, -0.0033231, -0.0011712, -0.0015078, 0.0015109
9: -0.0005337, 0.0020277, -0.0005058, 0.0019894, -0.0017520, 0.0017484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013674, upper bound: 0.0013385
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013569, upper bound: 0.0013404
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9894425, 0.9932938, 0.9894209, 0.9933371, -0.0026710, 0.0026837
1: -0.0038946, -0.0029350, -0.0039000, -0.0029242, -0.0006656, 0.0006687
2: 0.0054998, 0.0105854, 0.0054426, 0.0106138, -0.0035438, 0.0035271
3: -0.0060911, -0.0037764, -0.0061041, -0.0037503, -0.0016054, 0.0016130
4: 0.0015924, 0.0025767, 0.0015813, 0.0025822, -0.0006859, 0.0006827
5: 0.0058767, 0.0122731, 0.0058048, 0.0123089, -0.0044571, 0.0044361
6: -0.0015742, 0.0000493, -0.0015833, 0.0000675, -0.0011259, 0.0011313
7: -0.0072106, -0.0030102, -0.0072341, -0.0029629, -0.0029132, 0.0029269
8: -0.0033561, -0.0011472, -0.0033685, -0.0011223, -0.0015320, 0.0015392
9: -0.0005337, 0.0020277, -0.0005625, 0.0020421, -0.0017848, 0.0017764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013674, upper bound: 0.0013385
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013569, upper bound: 0.0013403
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9891263, 0.9931122, 0.9895107, 0.9932514, -0.0029842, 0.0024989
1: -0.0039734, -0.0029802, -0.0038776, -0.0029455, -0.0007436, 0.0006227
2: 0.0057396, 0.0110029, 0.0055557, 0.0104952, -0.0032997, 0.0039406
3: -0.0062812, -0.0038855, -0.0060501, -0.0038018, -0.0017936, 0.0015019
4: 0.0016388, 0.0026575, 0.0016032, 0.0025592, -0.0006387, 0.0007627
5: 0.0061784, 0.0127982, 0.0059470, 0.0121597, -0.0041502, 0.0049563
6: -0.0017075, -0.0000273, -0.0015454, 0.0000314, -0.0012580, 0.0010534
7: -0.0075554, -0.0032083, -0.0071361, -0.0030563, -0.0032547, 0.0027254
8: -0.0035375, -0.0012513, -0.0033170, -0.0011714, -0.0017116, 0.0014332
9: -0.0004129, 0.0022380, -0.0005055, 0.0019823, -0.0016619, 0.0019847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0013536
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013032, upper bound: 0.0013555
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9891263, 0.9931122, 0.9894315, 0.9933367, -0.0030342, 0.0025420
1: -0.0039734, -0.0029802, -0.0038973, -0.0029243, -0.0007560, 0.0006334
2: 0.0057396, 0.0110029, 0.0054432, 0.0105998, -0.0033567, 0.0040067
3: -0.0062812, -0.0038855, -0.0060977, -0.0037506, -0.0018237, 0.0015278
4: 0.0016388, 0.0026575, 0.0015814, 0.0025795, -0.0006497, 0.0007755
5: 0.0061784, 0.0127982, 0.0058055, 0.0122913, -0.0042218, 0.0050393
6: -0.0017075, -0.0000273, -0.0015788, 0.0000673, -0.0012790, 0.0010715
7: -0.0075554, -0.0032083, -0.0072225, -0.0029634, -0.0033093, 0.0027724
8: -0.0035375, -0.0012513, -0.0033624, -0.0011226, -0.0017403, 0.0014580
9: -0.0004129, 0.0022380, -0.0005622, 0.0020350, -0.0016906, 0.0020180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0013536
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013032, upper bound: 0.0013555
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9890873, 0.9932092, 0.9895000, 0.9932519, -0.0030807, 0.0025467
1: -0.0039831, -0.0029560, -0.0038803, -0.0029454, -0.0007676, 0.0006346
2: 0.0056114, 0.0110544, 0.0055551, 0.0105094, -0.0033629, 0.0040681
3: -0.0063046, -0.0038272, -0.0060565, -0.0038016, -0.0018516, 0.0015306
4: 0.0016140, 0.0026675, 0.0016031, 0.0025620, -0.0006509, 0.0007874
5: 0.0060171, 0.0128630, 0.0059463, 0.0121775, -0.0042296, 0.0051166
6: -0.0017239, 0.0000136, -0.0015499, 0.0000316, -0.0012986, 0.0010735
7: -0.0075980, -0.0031024, -0.0071478, -0.0030559, -0.0033600, 0.0027775
8: -0.0035599, -0.0011957, -0.0033231, -0.0011712, -0.0017670, 0.0014607
9: -0.0004774, 0.0022640, -0.0005058, 0.0019894, -0.0016937, 0.0020489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013436, upper bound: 0.0013605
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013478, upper bound: 0.0013555
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9890873, 0.9932092, 0.9894209, 0.9933371, -0.0031229, 0.0025960
1: -0.0039831, -0.0029560, -0.0039000, -0.0029242, -0.0007781, 0.0006469
2: 0.0056114, 0.0110544, 0.0054426, 0.0106138, -0.0034280, 0.0041237
3: -0.0063046, -0.0038272, -0.0061041, -0.0037503, -0.0018769, 0.0015603
4: 0.0016140, 0.0026675, 0.0015813, 0.0025822, -0.0006635, 0.0007981
5: 0.0060171, 0.0128630, 0.0058048, 0.0123089, -0.0043116, 0.0051865
6: -0.0017239, 0.0000136, -0.0015833, 0.0000675, -0.0013164, 0.0010943
7: -0.0075980, -0.0031024, -0.0072341, -0.0029629, -0.0034059, 0.0028313
8: -0.0035599, -0.0011957, -0.0033685, -0.0011223, -0.0017911, 0.0014890
9: -0.0004774, 0.0022640, -0.0005625, 0.0020421, -0.0017265, 0.0020769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013575, upper bound: 0.0013536
time: 1.14 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013478, upper bound: 0.0013555
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9894401, 0.9932002, 0.9893214, 0.9931608, -0.0027041, 0.0027607
1: -0.0038952, -0.0029583, -0.0039248, -0.0029681, -0.0006738, 0.0006879
2: 0.0056233, 0.0105885, 0.0056753, 0.0107453, -0.0036455, 0.0035708
3: -0.0060925, -0.0038326, -0.0061639, -0.0038563, -0.0016253, 0.0016593
4: 0.0016163, 0.0025773, 0.0016263, 0.0026076, -0.0007056, 0.0006911
5: 0.0060321, 0.0122770, 0.0060975, 0.0124742, -0.0045851, 0.0044911
6: -0.0015752, 0.0000098, -0.0016253, -0.0000068, -0.0011399, 0.0011637
7: -0.0072131, -0.0031122, -0.0073427, -0.0031552, -0.0029492, 0.0030110
8: -0.0033575, -0.0012008, -0.0034256, -0.0012234, -0.0015510, 0.0015834
9: -0.0004714, 0.0020293, -0.0004452, 0.0021083, -0.0018361, 0.0017984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013250, upper bound: 0.0012990
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0012990
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9894388, 0.9932013, 0.9893240, 0.9931329, -0.0027386, 0.0027938
1: -0.0038955, -0.0029580, -0.0039241, -0.0029750, -0.0006824, 0.0006961
2: 0.0056220, 0.0105901, 0.0057122, 0.0107418, -0.0036891, 0.0036163
3: -0.0060933, -0.0038320, -0.0061623, -0.0038731, -0.0016460, 0.0016791
4: 0.0016160, 0.0025776, 0.0016335, 0.0026069, -0.0007140, 0.0006999
5: 0.0060304, 0.0122791, 0.0061439, 0.0124698, -0.0046400, 0.0045484
6: -0.0015757, 0.0000103, -0.0016241, -0.0000186, -0.0011544, 0.0011777
7: -0.0072145, -0.0031111, -0.0073398, -0.0031856, -0.0029868, 0.0030470
8: -0.0033582, -0.0012002, -0.0034241, -0.0012394, -0.0015708, 0.0016024
9: -0.0004721, 0.0020301, -0.0004267, 0.0021065, -0.0018581, 0.0018214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013355, upper bound: 0.0012909
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0012909
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9894897, 0.9932194, 0.9891889, 0.9932372, -0.0027156, 0.0028848
1: -0.0038828, -0.0029535, -0.0039578, -0.0029491, -0.0006767, 0.0007188
2: 0.0055981, 0.0105230, 0.0055745, 0.0109202, -0.0038094, 0.0035859
3: -0.0060627, -0.0038211, -0.0062435, -0.0038104, -0.0016321, 0.0017339
4: 0.0016114, 0.0025646, 0.0016068, 0.0026415, -0.0007373, 0.0006940
5: 0.0060004, 0.0121947, 0.0059707, 0.0126942, -0.0047912, 0.0045101
6: -0.0015543, 0.0000179, -0.0016811, 0.0000254, -0.0011447, 0.0012161
7: -0.0071591, -0.0030914, -0.0074871, -0.0030719, -0.0029617, 0.0031463
8: -0.0033290, -0.0011899, -0.0035015, -0.0011796, -0.0015575, 0.0016546
9: -0.0004841, 0.0019963, -0.0004960, 0.0021964, -0.0019186, 0.0018060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013793, upper bound: 0.0012718
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0012718
time: 1.03 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9894897, 0.9931929, 0.9891903, 0.9932388, -0.0027089, 0.0029092
1: -0.0038828, -0.0029601, -0.0039574, -0.0029487, -0.0006750, 0.0007249
2: 0.0056329, 0.0105230, 0.0055725, 0.0109183, -0.0038416, 0.0035770
3: -0.0060627, -0.0038370, -0.0062427, -0.0038095, -0.0016281, 0.0017485
4: 0.0016181, 0.0025646, 0.0016064, 0.0026411, -0.0007435, 0.0006923
5: 0.0060442, 0.0121946, 0.0059682, 0.0126918, -0.0048317, 0.0044990
6: -0.0015543, 0.0000067, -0.0016805, 0.0000260, -0.0011419, 0.0012263
7: -0.0071590, -0.0031202, -0.0074855, -0.0030702, -0.0029544, 0.0031729
8: -0.0033290, -0.0012050, -0.0035007, -0.0011788, -0.0015537, 0.0016686
9: -0.0004666, 0.0019963, -0.0004970, 0.0021954, -0.0019348, 0.0018016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013727, upper bound: 0.0012735
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0012735
time: 1.01 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9894297, 0.9932008, 0.9892774, 0.9932582, -0.0027366, 0.0028488
1: -0.0038978, -0.0029581, -0.0039357, -0.0029438, -0.0006819, 0.0007098
2: 0.0056225, 0.0106022, 0.0055468, 0.0108034, -0.0037618, 0.0036137
3: -0.0060988, -0.0038323, -0.0061904, -0.0037978, -0.0016448, 0.0017122
4: 0.0016161, 0.0025799, 0.0016015, 0.0026189, -0.0007281, 0.0006994
5: 0.0060311, 0.0122943, 0.0059358, 0.0125473, -0.0047314, 0.0045451
6: -0.0015796, 0.0000101, -0.0016438, 0.0000343, -0.0011536, 0.0012009
7: -0.0072245, -0.0031116, -0.0073906, -0.0030490, -0.0029847, 0.0031070
8: -0.0033634, -0.0012005, -0.0034508, -0.0011676, -0.0015696, 0.0016340
9: -0.0004718, 0.0020362, -0.0005100, 0.0021375, -0.0018947, 0.0018201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013250, upper bound: 0.0013483
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0013482
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9894289, 0.9932018, 0.9892794, 0.9932292, -0.0027710, 0.0028307
1: -0.0038980, -0.0029579, -0.0039352, -0.0029511, -0.0006905, 0.0007053
2: 0.0056212, 0.0106034, 0.0055851, 0.0108006, -0.0037379, 0.0036591
3: -0.0060993, -0.0038316, -0.0061891, -0.0038152, -0.0016654, 0.0017013
4: 0.0016159, 0.0025802, 0.0016089, 0.0026183, -0.0007235, 0.0007082
5: 0.0060294, 0.0122957, 0.0059841, 0.0125437, -0.0047012, 0.0046021
6: -0.0015800, 0.0000105, -0.0016429, 0.0000220, -0.0011681, 0.0011932
7: -0.0072254, -0.0031105, -0.0073883, -0.0030807, -0.0030222, 0.0030872
8: -0.0033639, -0.0011999, -0.0034496, -0.0011843, -0.0015893, 0.0016235
9: -0.0004725, 0.0020368, -0.0004907, 0.0021361, -0.0018826, 0.0018429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013355, upper bound: 0.0013386
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0013386
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9894797, 0.9932199, 0.9891454, 0.9933388, -0.0027699, 0.0029240
1: -0.0038853, -0.0029534, -0.0039686, -0.0029238, -0.0006902, 0.0007286
2: 0.0055974, 0.0105363, 0.0054404, 0.0109776, -0.0038611, 0.0036576
3: -0.0060688, -0.0038208, -0.0062697, -0.0037494, -0.0016648, 0.0017574
4: 0.0016112, 0.0025672, 0.0015809, 0.0026526, -0.0007473, 0.0007079
5: 0.0059995, 0.0122113, 0.0058020, 0.0127664, -0.0048562, 0.0046004
6: -0.0015585, 0.0000181, -0.0016994, 0.0000682, -0.0011676, 0.0012326
7: -0.0071700, -0.0030908, -0.0075346, -0.0029611, -0.0030210, 0.0031890
8: -0.0033348, -0.0011896, -0.0035265, -0.0011214, -0.0015887, 0.0016771
9: -0.0004845, 0.0020030, -0.0005635, 0.0022253, -0.0019446, 0.0018422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013096
time: 0.99 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013138
time: 0.88 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9894806, 0.9931935, 0.9891462, 0.9933404, -0.0027666, 0.0029489
1: -0.0038851, -0.0029600, -0.0039684, -0.0029234, -0.0006894, 0.0007348
2: 0.0056322, 0.0105351, 0.0054382, 0.0109766, -0.0038939, 0.0036533
3: -0.0060682, -0.0038367, -0.0062692, -0.0037484, -0.0016628, 0.0017723
4: 0.0016180, 0.0025669, 0.0015804, 0.0026524, -0.0007537, 0.0007071
5: 0.0060433, 0.0122098, 0.0057993, 0.0127651, -0.0048975, 0.0045949
6: -0.0015582, 0.0000070, -0.0016991, 0.0000689, -0.0011662, 0.0012430
7: -0.0071690, -0.0031195, -0.0075337, -0.0029593, -0.0030174, 0.0032161
8: -0.0033343, -0.0012047, -0.0035260, -0.0011204, -0.0015868, 0.0016913
9: -0.0004670, 0.0020024, -0.0005647, 0.0022248, -0.0019612, 0.0018400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013771, upper bound: 0.0013133
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0013167
time: 0.92 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9894355, 0.9933180, 0.9895039, 0.9933872, -0.0026911, 0.0025679
1: -0.0038963, -0.0029289, -0.0038793, -0.0029117, -0.0006705, 0.0006399
2: 0.0054679, 0.0105945, 0.0053764, 0.0105042, -0.0033909, 0.0035536
3: -0.0060953, -0.0037619, -0.0060542, -0.0037202, -0.0016174, 0.0015434
4: 0.0015862, 0.0025784, 0.0015685, 0.0025609, -0.0006563, 0.0006878
5: 0.0058366, 0.0122845, 0.0057216, 0.0121709, -0.0042649, 0.0044694
6: -0.0015771, 0.0000594, -0.0015483, 0.0000886, -0.0011344, 0.0010825
7: -0.0072181, -0.0029838, -0.0071435, -0.0029083, -0.0029350, 0.0028007
8: -0.0033601, -0.0011333, -0.0033208, -0.0010936, -0.0015435, 0.0014728
9: -0.0005497, 0.0020323, -0.0005958, 0.0019868, -0.0017078, 0.0017898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
time: 0.85 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9894355, 0.9933180, 0.9894122, 0.9933609, -0.0026402, 0.0026779
1: -0.0038963, -0.0029289, -0.0039021, -0.0029182, -0.0006579, 0.0006673
2: 0.0054679, 0.0105945, 0.0054112, 0.0106253, -0.0035362, 0.0034864
3: -0.0060953, -0.0037619, -0.0061093, -0.0037361, -0.0015869, 0.0016095
4: 0.0015862, 0.0025784, 0.0015752, 0.0025844, -0.0006844, 0.0006748
5: 0.0058366, 0.0122845, 0.0057653, 0.0123233, -0.0044475, 0.0043850
6: -0.0015771, 0.0000594, -0.0015870, 0.0000775, -0.0011129, 0.0011288
7: -0.0072181, -0.0029838, -0.0072435, -0.0029370, -0.0028795, 0.0029206
8: -0.0033601, -0.0011333, -0.0033735, -0.0011087, -0.0015143, 0.0015359
9: -0.0005497, 0.0020323, -0.0005783, 0.0020478, -0.0017810, 0.0017559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
time: 0.84 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9894249, 0.9933185, 0.9895304, 0.9933859, -0.0026617, 0.0025622
1: -0.0038990, -0.0029288, -0.0038727, -0.0029120, -0.0006632, 0.0006384
2: 0.0054672, 0.0106086, 0.0053781, 0.0104693, -0.0033833, 0.0035147
3: -0.0061017, -0.0037616, -0.0060383, -0.0037210, -0.0015997, 0.0015399
4: 0.0015861, 0.0025812, 0.0015688, 0.0025542, -0.0006548, 0.0006803
5: 0.0058358, 0.0123023, 0.0057237, 0.0121270, -0.0042553, 0.0044205
6: -0.0015816, 0.0000596, -0.0015371, 0.0000881, -0.0011220, 0.0010800
7: -0.0072298, -0.0029833, -0.0071147, -0.0029097, -0.0029029, 0.0027944
8: -0.0033662, -0.0011330, -0.0033057, -0.0010943, -0.0015266, 0.0014695
9: -0.0005500, 0.0020394, -0.0005949, 0.0019693, -0.0017040, 0.0017702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
time: 0.90 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9894249, 0.9933185, 0.9894527, 0.9934784, -0.0027039, 0.0026020
1: -0.0038990, -0.0029288, -0.0038921, -0.0028889, -0.0006737, 0.0006484
2: 0.0054672, 0.0106086, 0.0052559, 0.0105719, -0.0034359, 0.0035705
3: -0.0061017, -0.0037616, -0.0060850, -0.0036654, -0.0016251, 0.0015639
4: 0.0015861, 0.0025812, 0.0015452, 0.0025741, -0.0006650, 0.0006911
5: 0.0058358, 0.0123023, 0.0055700, 0.0122561, -0.0043215, 0.0044907
6: -0.0015816, 0.0000596, -0.0015699, 0.0001271, -0.0011398, 0.0010968
7: -0.0072298, -0.0029833, -0.0071995, -0.0028088, -0.0029490, 0.0028379
8: -0.0033662, -0.0011330, -0.0033503, -0.0010413, -0.0015508, 0.0014924
9: -0.0005500, 0.0020394, -0.0006565, 0.0020209, -0.0017305, 0.0017983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
time: 0.90 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9894355, 0.9933180, 0.9892465, 0.9931781, -0.0025530, 0.0028895
1: -0.0038963, -0.0029289, -0.0039434, -0.0029638, -0.0006361, 0.0007200
2: 0.0054679, 0.0105945, 0.0056525, 0.0108441, -0.0038155, 0.0033711
3: -0.0060953, -0.0037619, -0.0062089, -0.0038459, -0.0015344, 0.0017367
4: 0.0015862, 0.0025784, 0.0016219, 0.0026267, -0.0007385, 0.0006525
5: 0.0058366, 0.0122845, 0.0060688, 0.0125985, -0.0047989, 0.0042400
6: -0.0015771, 0.0000594, -0.0016568, 0.0000005, -0.0010762, 0.0012180
7: -0.0072181, -0.0029838, -0.0074243, -0.0031363, -0.0027844, 0.0031514
8: -0.0033601, -0.0011333, -0.0034685, -0.0012135, -0.0014643, 0.0016573
9: -0.0005497, 0.0020323, -0.0004567, 0.0021580, -0.0019217, 0.0016979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013808
time: 0.82 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013809
time: 1.07 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9894355, 0.9933180, 0.9891647, 0.9932756, -0.0026010, 0.0029321
1: -0.0038963, -0.0029289, -0.0039638, -0.0029395, -0.0006481, 0.0007306
2: 0.0054679, 0.0105945, 0.0055239, 0.0109522, -0.0038718, 0.0034346
3: -0.0060953, -0.0037619, -0.0062581, -0.0037874, -0.0015633, 0.0017623
4: 0.0015862, 0.0025784, 0.0015970, 0.0026477, -0.0007494, 0.0006648
5: 0.0058366, 0.0122845, 0.0059071, 0.0127345, -0.0048697, 0.0043198
6: -0.0015771, 0.0000594, -0.0016913, 0.0000415, -0.0010964, 0.0012360
7: -0.0072181, -0.0029838, -0.0075136, -0.0030301, -0.0028367, 0.0031978
8: -0.0033601, -0.0011333, -0.0035155, -0.0011577, -0.0014918, 0.0016817
9: -0.0005497, 0.0020323, -0.0005215, 0.0022125, -0.0019500, 0.0017298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013808
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013809
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9894249, 0.9933185, 0.9892024, 0.9932750, -0.0026091, 0.0029285
1: -0.0038990, -0.0029288, -0.0039544, -0.0029397, -0.0006501, 0.0007297
2: 0.0054672, 0.0106086, 0.0055247, 0.0109023, -0.0038670, 0.0034452
3: -0.0061017, -0.0037616, -0.0062354, -0.0037877, -0.0015681, 0.0017601
4: 0.0015861, 0.0025812, 0.0015972, 0.0026380, -0.0007485, 0.0006668
5: 0.0058358, 0.0123023, 0.0059080, 0.0126717, -0.0048637, 0.0043332
6: -0.0015816, 0.0000596, -0.0016754, 0.0000413, -0.0010998, 0.0012345
7: -0.0072298, -0.0029833, -0.0074723, -0.0030307, -0.0028456, 0.0031939
8: -0.0033662, -0.0011330, -0.0034938, -0.0011580, -0.0014964, 0.0016797
9: -0.0005500, 0.0020394, -0.0005211, 0.0021873, -0.0019476, 0.0017352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
time: 0.82 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9894249, 0.9933185, 0.9891208, 0.9933779, -0.0026478, 0.0029676
1: -0.0038990, -0.0029288, -0.0039747, -0.0029140, -0.0006598, 0.0007395
2: 0.0054672, 0.0106086, 0.0053887, 0.0110101, -0.0039187, 0.0034964
3: -0.0061017, -0.0037616, -0.0062844, -0.0037258, -0.0015914, 0.0017836
4: 0.0015861, 0.0025812, 0.0015709, 0.0026589, -0.0007585, 0.0006767
5: 0.0058358, 0.0123023, 0.0057370, 0.0128072, -0.0049287, 0.0043976
6: -0.0015816, 0.0000596, -0.0017098, 0.0000847, -0.0011162, 0.0012510
7: -0.0072298, -0.0029833, -0.0075614, -0.0029184, -0.0028878, 0.0032366
8: -0.0033662, -0.0011330, -0.0035406, -0.0010989, -0.0015187, 0.0017021
9: -0.0005500, 0.0020394, -0.0005896, 0.0022416, -0.0019737, 0.0017610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014223
time: 0.87 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9891739, 0.9932092, 0.9895159, 0.9932218, -0.0028530, 0.0026918
1: -0.0039615, -0.0029560, -0.0038763, -0.0029529, -0.0007109, 0.0006707
2: 0.0056115, 0.0109400, 0.0055949, 0.0104885, -0.0035545, 0.0037674
3: -0.0062525, -0.0038272, -0.0060470, -0.0038197, -0.0017147, 0.0016178
4: 0.0016140, 0.0026453, 0.0016108, 0.0025579, -0.0006880, 0.0007292
5: 0.0060172, 0.0127190, 0.0059963, 0.0121512, -0.0044706, 0.0047384
6: -0.0016874, 0.0000136, -0.0015433, 0.0000189, -0.0012027, 0.0011347
7: -0.0075034, -0.0031024, -0.0071305, -0.0030887, -0.0031116, 0.0029358
8: -0.0035101, -0.0011957, -0.0033140, -0.0011885, -0.0016364, 0.0015439
9: -0.0004774, 0.0022063, -0.0004858, 0.0019789, -0.0017902, 0.0018975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013045, upper bound: 0.0014206
time: 0.88 seconds

## Relational analysis of IS_A2_A1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013050, upper bound: 0.0014091
time: 0.88 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9891739, 0.9932092, 0.9894366, 0.9933152, -0.0029126, 0.0027339
1: -0.0039615, -0.0029560, -0.0038961, -0.0029296, -0.0007257, 0.0006812
2: 0.0056115, 0.0109400, 0.0054715, 0.0105931, -0.0036101, 0.0038460
3: -0.0062525, -0.0038272, -0.0060947, -0.0037635, -0.0017506, 0.0016432
4: 0.0016140, 0.0026453, 0.0015869, 0.0025782, -0.0006987, 0.0007444
5: 0.0060172, 0.0127190, 0.0058412, 0.0122828, -0.0045406, 0.0048373
6: -0.0016874, 0.0000136, -0.0015767, 0.0000583, -0.0012278, 0.0011525
7: -0.0075034, -0.0031024, -0.0072170, -0.0029868, -0.0031766, 0.0029818
8: -0.0035101, -0.0011957, -0.0033595, -0.0011349, -0.0016705, 0.0015681
9: -0.0004774, 0.0022063, -0.0005479, 0.0020316, -0.0018183, 0.0019371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0013918
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013050, upper bound: 0.0014091
time: 0.83 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9891739, 0.9932092, 0.9894181, 0.9932045, -0.0028482, 0.0027991
1: -0.0039615, -0.0029560, -0.0039007, -0.0029572, -0.0007097, 0.0006975
2: 0.0056115, 0.0109400, 0.0056176, 0.0106175, -0.0036962, 0.0037610
3: -0.0062525, -0.0038272, -0.0061058, -0.0038300, -0.0017119, 0.0016823
4: 0.0016140, 0.0026453, 0.0016152, 0.0025829, -0.0007154, 0.0007279
5: 0.0060172, 0.0127190, 0.0060250, 0.0123135, -0.0046488, 0.0047304
6: -0.0016874, 0.0000136, -0.0015845, 0.0000116, -0.0012006, 0.0011799
7: -0.0075034, -0.0031024, -0.0072371, -0.0031075, -0.0031064, 0.0030528
8: -0.0035101, -0.0011957, -0.0033701, -0.0011984, -0.0016336, 0.0016054
9: -0.0004774, 0.0022063, -0.0004743, 0.0020439, -0.0018616, 0.0018943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013140, upper bound: 0.0014166
time: 0.92 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013143, upper bound: 0.0014068
time: 0.92 seconds

## BFS IS instance: IS_A2_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9891739, 0.9932092, 0.9893442, 0.9932998, -0.0029106, 0.0028435
1: -0.0039615, -0.0029560, -0.0039191, -0.0029335, -0.0007252, 0.0007085
2: 0.0056115, 0.0109400, 0.0054918, 0.0107152, -0.0037548, 0.0038434
3: -0.0062525, -0.0038272, -0.0061502, -0.0037728, -0.0017493, 0.0017090
4: 0.0016140, 0.0026453, 0.0015908, 0.0026018, -0.0007267, 0.0007439
5: 0.0060172, 0.0127190, 0.0058667, 0.0124364, -0.0047226, 0.0048339
6: -0.0016874, 0.0000136, -0.0016157, 0.0000518, -0.0012269, 0.0011986
7: -0.0075034, -0.0031024, -0.0073178, -0.0030036, -0.0031744, 0.0031013
8: -0.0035101, -0.0011957, -0.0034125, -0.0011437, -0.0016694, 0.0016309
9: -0.0004774, 0.0022063, -0.0005377, 0.0020931, -0.0018911, 0.0019357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013189, upper bound: 0.0013901
time: 0.88 seconds

## Relational analysis of IS_A2_A1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013143, upper bound: 0.0014068
time: 1.22 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9891304, 0.9933133, 0.9895053, 0.9932222, -0.0028935, 0.0027431
1: -0.0039724, -0.0029301, -0.0038790, -0.0029528, -0.0007210, 0.0006835
2: 0.0054740, 0.0109975, 0.0055944, 0.0105024, -0.0036222, 0.0038208
3: -0.0062787, -0.0037646, -0.0060534, -0.0038194, -0.0017391, 0.0016487
4: 0.0015874, 0.0026564, 0.0016107, 0.0025606, -0.0007011, 0.0007395
5: 0.0058443, 0.0127914, 0.0059957, 0.0121687, -0.0045558, 0.0048056
6: -0.0017058, 0.0000575, -0.0015477, 0.0000191, -0.0012197, 0.0011563
7: -0.0075509, -0.0029889, -0.0071420, -0.0030883, -0.0031558, 0.0029917
8: -0.0035351, -0.0011360, -0.0033201, -0.0011883, -0.0016596, 0.0015733
9: -0.0005466, 0.0022353, -0.0004860, 0.0019859, -0.0018243, 0.0019244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013466, upper bound: 0.0014206
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013475, upper bound: 0.0014091
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9891304, 0.9933133, 0.9894260, 0.9933156, -0.0029475, 0.0027597
1: -0.0039724, -0.0029301, -0.0038987, -0.0029295, -0.0007345, 0.0006876
2: 0.0054740, 0.0109975, 0.0054709, 0.0106072, -0.0036442, 0.0038922
3: -0.0062787, -0.0037646, -0.0061011, -0.0037632, -0.0017716, 0.0016587
4: 0.0015874, 0.0026564, 0.0015868, 0.0025809, -0.0007053, 0.0007533
5: 0.0058443, 0.0127914, 0.0058404, 0.0123005, -0.0045834, 0.0048954
6: -0.0017058, 0.0000575, -0.0015812, 0.0000585, -0.0012425, 0.0011633
7: -0.0075509, -0.0029889, -0.0072286, -0.0029863, -0.0032147, 0.0030099
8: -0.0035351, -0.0011360, -0.0033656, -0.0011346, -0.0016906, 0.0015829
9: -0.0005466, 0.0022353, -0.0005482, 0.0020387, -0.0018354, 0.0019603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013525, upper bound: 0.0013918
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013475, upper bound: 0.0014091
time: 0.83 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9891304, 0.9933133, 0.9894079, 0.9932051, -0.0028890, 0.0028533
1: -0.0039724, -0.0029301, -0.0039032, -0.0029571, -0.0007199, 0.0007110
2: 0.0054740, 0.0109975, 0.0056169, 0.0106310, -0.0037678, 0.0038149
3: -0.0062787, -0.0037646, -0.0061119, -0.0038297, -0.0017364, 0.0017149
4: 0.0015874, 0.0026564, 0.0016150, 0.0025855, -0.0007293, 0.0007384
5: 0.0058443, 0.0127914, 0.0060240, 0.0123305, -0.0047389, 0.0047981
6: -0.0017058, 0.0000575, -0.0015888, 0.0000119, -0.0012178, 0.0012028
7: -0.0075509, -0.0029889, -0.0072483, -0.0031069, -0.0031508, 0.0031120
8: -0.0035351, -0.0011360, -0.0033759, -0.0011980, -0.0016570, 0.0016366
9: -0.0005466, 0.0022353, -0.0004747, 0.0020507, -0.0018977, 0.0019214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014166
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014068
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9891304, 0.9933133, 0.9893340, 0.9933004, -0.0029458, 0.0028702
1: -0.0039724, -0.0029301, -0.0039216, -0.0029333, -0.0007340, 0.0007152
2: 0.0054740, 0.0109975, 0.0054910, 0.0107285, -0.0037900, 0.0038899
3: -0.0062787, -0.0037646, -0.0061563, -0.0037724, -0.0017705, 0.0017251
4: 0.0015874, 0.0026564, 0.0015907, 0.0026044, -0.0007336, 0.0007529
5: 0.0058443, 0.0127914, 0.0058656, 0.0124531, -0.0047668, 0.0048925
6: -0.0017058, 0.0000575, -0.0016199, 0.0000521, -0.0012418, 0.0012099
7: -0.0075509, -0.0029889, -0.0073288, -0.0030029, -0.0032128, 0.0031303
8: -0.0035351, -0.0011360, -0.0034183, -0.0011433, -0.0016896, 0.0016462
9: -0.0005466, 0.0022353, -0.0005381, 0.0020998, -0.0019089, 0.0019592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A1_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013605, upper bound: 0.0013901
time: 0.90 seconds

## Relational analysis of IS_A2_A1_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014068
time: 0.88 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9894134, 0.9932876, 0.9895107, 0.9932514, -0.0026111, 0.0026584
1: -0.0039018, -0.0029365, -0.0038776, -0.0029455, -0.0006506, 0.0006624
2: 0.0055080, 0.0106237, 0.0055557, 0.0104952, -0.0035103, 0.0034479
3: -0.0061086, -0.0037801, -0.0060501, -0.0038018, -0.0015694, 0.0015978
4: 0.0015939, 0.0025841, 0.0016032, 0.0025592, -0.0006794, 0.0006673
5: 0.0058871, 0.0123213, 0.0059470, 0.0121597, -0.0044151, 0.0043366
6: -0.0015864, 0.0000466, -0.0015454, 0.0000314, -0.0011007, 0.0011206
7: -0.0072422, -0.0030170, -0.0071361, -0.0030563, -0.0028478, 0.0028993
8: -0.0033728, -0.0011507, -0.0033170, -0.0011714, -0.0014976, 0.0015247
9: -0.0005295, 0.0020470, -0.0005055, 0.0019823, -0.0017680, 0.0017366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013010, upper bound: 0.0013788
time: 0.90 seconds

## Relational analysis of IS_A2_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012990, upper bound: 0.0013940
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9894134, 0.9932876, 0.9894315, 0.9933367, -0.0026827, 0.0026914
1: -0.0039018, -0.0029365, -0.0038973, -0.0029243, -0.0006685, 0.0006706
2: 0.0055080, 0.0106237, 0.0054432, 0.0105998, -0.0035539, 0.0035424
3: -0.0061086, -0.0037801, -0.0060977, -0.0037506, -0.0016124, 0.0016176
4: 0.0015939, 0.0025841, 0.0015814, 0.0025795, -0.0006879, 0.0006856
5: 0.0058871, 0.0123213, 0.0058055, 0.0122913, -0.0044699, 0.0044555
6: -0.0015864, 0.0000466, -0.0015788, 0.0000673, -0.0011308, 0.0011345
7: -0.0072422, -0.0030170, -0.0072225, -0.0029634, -0.0029258, 0.0029353
8: -0.0033728, -0.0011507, -0.0033624, -0.0011226, -0.0015387, 0.0015437
9: -0.0005295, 0.0020470, -0.0005622, 0.0020350, -0.0017899, 0.0017842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013010, upper bound: 0.0013787
time: 0.88 seconds

## Relational analysis of IS_A2_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012990, upper bound: 0.0013940
time: 0.87 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9893707, 0.9933856, 0.9895000, 0.9932519, -0.0026537, 0.0027098
1: -0.0039125, -0.0029121, -0.0038803, -0.0029454, -0.0006612, 0.0006752
2: 0.0053786, 0.0106801, 0.0055551, 0.0105094, -0.0035782, 0.0035042
3: -0.0061342, -0.0037212, -0.0060565, -0.0038016, -0.0015949, 0.0016286
4: 0.0015689, 0.0025950, 0.0016031, 0.0025620, -0.0006926, 0.0006782
5: 0.0057243, 0.0123922, 0.0059463, 0.0121775, -0.0045004, 0.0044073
6: -0.0016044, 0.0000880, -0.0015499, 0.0000316, -0.0011186, 0.0011423
7: -0.0072888, -0.0029101, -0.0071478, -0.0030559, -0.0028942, 0.0029554
8: -0.0033973, -0.0010945, -0.0033231, -0.0011712, -0.0015220, 0.0015542
9: -0.0005947, 0.0020754, -0.0005058, 0.0019894, -0.0018022, 0.0017649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014067
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013403, upper bound: 0.0013939
time: 0.88 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9893707, 0.9933856, 0.9894209, 0.9933371, -0.0027242, 0.0027203
1: -0.0039125, -0.0029121, -0.0039000, -0.0029242, -0.0006788, 0.0006778
2: 0.0053786, 0.0106801, 0.0054426, 0.0106138, -0.0035922, 0.0035973
3: -0.0061342, -0.0037212, -0.0061041, -0.0037503, -0.0016373, 0.0016350
4: 0.0015689, 0.0025950, 0.0015813, 0.0025822, -0.0006953, 0.0006962
5: 0.0057243, 0.0123922, 0.0058048, 0.0123089, -0.0045180, 0.0045244
6: -0.0016044, 0.0000880, -0.0015833, 0.0000675, -0.0011483, 0.0011467
7: -0.0072888, -0.0029101, -0.0072341, -0.0029629, -0.0029711, 0.0029669
8: -0.0033973, -0.0010945, -0.0033685, -0.0011223, -0.0015625, 0.0015603
9: -0.0005947, 0.0020754, -0.0005625, 0.0020421, -0.0018092, 0.0018118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013366, upper bound: 0.0014011
time: 0.93 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013430, upper bound: 0.0014011
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9890425, 0.9932153, 0.9895107, 0.9932514, -0.0030583, 0.0025549
1: -0.0039943, -0.0029545, -0.0038776, -0.0029455, -0.0007621, 0.0006366
2: 0.0056035, 0.0111135, 0.0055557, 0.0104952, -0.0033737, 0.0040385
3: -0.0063315, -0.0038236, -0.0060501, -0.0038018, -0.0018381, 0.0015356
4: 0.0016124, 0.0026789, 0.0016032, 0.0025592, -0.0006530, 0.0007816
5: 0.0060072, 0.0129373, 0.0059470, 0.0121597, -0.0042433, 0.0050794
6: -0.0017428, 0.0000161, -0.0015454, 0.0000314, -0.0012892, 0.0010770
7: -0.0076468, -0.0030959, -0.0071361, -0.0030563, -0.0033355, 0.0027865
8: -0.0035855, -0.0011922, -0.0033170, -0.0011714, -0.0017541, 0.0014654
9: -0.0004814, 0.0022937, -0.0005055, 0.0019823, -0.0016992, 0.0020340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014299
time: 0.89 seconds

## Relational analysis of IS_A2_A2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014185
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9890425, 0.9932153, 0.9894315, 0.9933367, -0.0031403, 0.0025839
1: -0.0039943, -0.0029545, -0.0038973, -0.0029243, -0.0007825, 0.0006438
2: 0.0056035, 0.0111135, 0.0054432, 0.0105998, -0.0034120, 0.0041467
3: -0.0063315, -0.0038236, -0.0060977, -0.0037506, -0.0018874, 0.0015530
4: 0.0016124, 0.0026789, 0.0015814, 0.0025795, -0.0006604, 0.0008026
5: 0.0060072, 0.0129373, 0.0058055, 0.0122913, -0.0042914, 0.0052155
6: -0.0017428, 0.0000161, -0.0015788, 0.0000673, -0.0013237, 0.0010892
7: -0.0076468, -0.0030959, -0.0072225, -0.0029634, -0.0034249, 0.0028181
8: -0.0035855, -0.0011922, -0.0033624, -0.0011226, -0.0018011, 0.0014820
9: -0.0004814, 0.0022937, -0.0005622, 0.0020350, -0.0017185, 0.0020885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012925, upper bound: 0.0013971
time: 0.90 seconds

## Relational analysis of IS_A2_A2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014185
time: 1.09 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9890040, 0.9933147, 0.9895000, 0.9932519, -0.0030967, 0.0026216
1: -0.0040039, -0.0029298, -0.0038803, -0.0029454, -0.0007716, 0.0006532
2: 0.0054722, 0.0111644, 0.0055551, 0.0105094, -0.0034619, 0.0040891
3: -0.0063547, -0.0037638, -0.0060565, -0.0038016, -0.0018612, 0.0015757
4: 0.0015870, 0.0026887, 0.0016031, 0.0025620, -0.0006700, 0.0007914
5: 0.0058420, 0.0130013, 0.0059463, 0.0121775, -0.0043541, 0.0051430
6: -0.0017590, 0.0000581, -0.0015499, 0.0000316, -0.0013054, 0.0011051
7: -0.0076888, -0.0029874, -0.0071478, -0.0030559, -0.0033773, 0.0028593
8: -0.0036076, -0.0011352, -0.0033231, -0.0011712, -0.0017761, 0.0015037
9: -0.0005476, 0.0023194, -0.0005058, 0.0019894, -0.0017436, 0.0020595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013306, upper bound: 0.0014299
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013318, upper bound: 0.0014185
time: 0.90 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9890040, 0.9933147, 0.9894209, 0.9933371, -0.0031768, 0.0026306
1: -0.0040039, -0.0029298, -0.0039000, -0.0029242, -0.0007916, 0.0006555
2: 0.0054722, 0.0111644, 0.0054426, 0.0106138, -0.0034737, 0.0041949
3: -0.0063547, -0.0037638, -0.0061041, -0.0037503, -0.0019093, 0.0015811
4: 0.0015870, 0.0026887, 0.0015813, 0.0025822, -0.0006723, 0.0008119
5: 0.0058420, 0.0130013, 0.0058048, 0.0123089, -0.0043690, 0.0052761
6: -0.0017590, 0.0000581, -0.0015833, 0.0000675, -0.0013391, 0.0011089
7: -0.0076888, -0.0029874, -0.0072341, -0.0029629, -0.0034647, 0.0028691
8: -0.0036076, -0.0011352, -0.0033685, -0.0011223, -0.0018221, 0.0015088
9: -0.0005476, 0.0023194, -0.0005625, 0.0020421, -0.0017495, 0.0021128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013253, upper bound: 0.0014178
time: 0.92 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013304, upper bound: 0.0014178
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9893652, 0.9932950, 0.9893214, 0.9931608, -0.0027369, 0.0028249
1: -0.0039139, -0.0029346, -0.0039248, -0.0029681, -0.0006820, 0.0007039
2: 0.0054981, 0.0106875, 0.0056753, 0.0107453, -0.0037303, 0.0036140
3: -0.0061376, -0.0037756, -0.0061639, -0.0038563, -0.0016449, 0.0016979
4: 0.0015920, 0.0025964, 0.0016263, 0.0026076, -0.0007220, 0.0006995
5: 0.0058746, 0.0124015, 0.0060975, 0.0124742, -0.0046917, 0.0045455
6: -0.0016068, 0.0000498, -0.0016253, -0.0000068, -0.0011537, 0.0011908
7: -0.0072949, -0.0030088, -0.0073427, -0.0031552, -0.0029849, 0.0030810
8: -0.0034005, -0.0011464, -0.0034256, -0.0012234, -0.0015697, 0.0016203
9: -0.0005345, 0.0020792, -0.0004452, 0.0021083, -0.0018788, 0.0018202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013381
time: 1.06 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013381
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9893656, 0.9932967, 0.9893240, 0.9931329, -0.0027672, 0.0028639
1: -0.0039138, -0.0029342, -0.0039241, -0.0029750, -0.0006895, 0.0007136
2: 0.0054959, 0.0106870, 0.0057122, 0.0107418, -0.0037817, 0.0036540
3: -0.0061374, -0.0037746, -0.0061623, -0.0038731, -0.0016631, 0.0017213
4: 0.0015916, 0.0025963, 0.0016335, 0.0026069, -0.0007319, 0.0007072
5: 0.0058719, 0.0124009, 0.0061439, 0.0124698, -0.0047564, 0.0045958
6: -0.0016067, 0.0000505, -0.0016241, -0.0000186, -0.0011665, 0.0012072
7: -0.0072945, -0.0030070, -0.0073398, -0.0031856, -0.0030180, 0.0031235
8: -0.0034003, -0.0011455, -0.0034241, -0.0012394, -0.0015871, 0.0016426
9: -0.0005356, 0.0020789, -0.0004267, 0.0021065, -0.0019047, 0.0018404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013294
time: 1.00 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013295
time: 1.07 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9893652, 0.9932950, 0.9892395, 0.9932599, -0.0027950, 0.0028928
1: -0.0039139, -0.0029346, -0.0039452, -0.0029434, -0.0006965, 0.0007208
2: 0.0054981, 0.0106875, 0.0055444, 0.0108536, -0.0038199, 0.0036908
3: -0.0061376, -0.0037756, -0.0062132, -0.0037967, -0.0016799, 0.0017387
4: 0.0015920, 0.0025964, 0.0016010, 0.0026286, -0.0007393, 0.0007144
5: 0.0058746, 0.0124015, 0.0059329, 0.0126104, -0.0048045, 0.0046421
6: -0.0016068, 0.0000498, -0.0016598, 0.0000350, -0.0011782, 0.0012194
7: -0.0072949, -0.0030088, -0.0074321, -0.0030471, -0.0030484, 0.0031550
8: -0.0034005, -0.0011464, -0.0034726, -0.0011666, -0.0016031, 0.0016592
9: -0.0005345, 0.0020792, -0.0005112, 0.0021628, -0.0019239, 0.0018589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013381
time: 0.85 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013381
time: 0.94 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9893656, 0.9932967, 0.9892458, 0.9932315, -0.0028373, 0.0028716
1: -0.0039138, -0.0029342, -0.0039436, -0.0029505, -0.0007070, 0.0007155
2: 0.0054959, 0.0106870, 0.0055822, 0.0108452, -0.0037919, 0.0037466
3: -0.0061374, -0.0037746, -0.0062094, -0.0038139, -0.0017053, 0.0017259
4: 0.0015916, 0.0025963, 0.0016083, 0.0026269, -0.0007339, 0.0007251
5: 0.0058719, 0.0124009, 0.0059803, 0.0125998, -0.0047692, 0.0047123
6: -0.0016067, 0.0000505, -0.0016571, 0.0000230, -0.0011960, 0.0012105
7: -0.0072945, -0.0030070, -0.0074251, -0.0030782, -0.0030945, 0.0031319
8: -0.0034003, -0.0011455, -0.0034689, -0.0011830, -0.0016274, 0.0016470
9: -0.0005356, 0.0020789, -0.0004922, 0.0021586, -0.0019098, 0.0018870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013295
time: 0.89 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013295
time: 0.90 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9893550, 0.9932957, 0.9892774, 0.9932582, -0.0027730, 0.0029032
1: -0.0039164, -0.0029345, -0.0039357, -0.0029438, -0.0006910, 0.0007234
2: 0.0054973, 0.0107009, 0.0055468, 0.0108034, -0.0038337, 0.0036618
3: -0.0061437, -0.0037752, -0.0061904, -0.0037978, -0.0016667, 0.0017449
4: 0.0015919, 0.0025990, 0.0016015, 0.0026189, -0.0007420, 0.0007087
5: 0.0058736, 0.0124183, 0.0059358, 0.0125473, -0.0048218, 0.0046055
6: -0.0016111, 0.0000501, -0.0016438, 0.0000343, -0.0011689, 0.0012238
7: -0.0073060, -0.0030081, -0.0073906, -0.0030490, -0.0030244, 0.0031664
8: -0.0034063, -0.0011461, -0.0034508, -0.0011676, -0.0015905, 0.0016652
9: -0.0005349, 0.0020859, -0.0005100, 0.0021375, -0.0019309, 0.0018443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013839
time: 0.98 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013839
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9893555, 0.9932973, 0.9892794, 0.9932292, -0.0028033, 0.0029008
1: -0.0039163, -0.0029341, -0.0039352, -0.0029511, -0.0006985, 0.0007228
2: 0.0054951, 0.0107002, 0.0055851, 0.0108006, -0.0038305, 0.0037017
3: -0.0061434, -0.0037743, -0.0061891, -0.0038152, -0.0016849, 0.0017435
4: 0.0015915, 0.0025989, 0.0016089, 0.0026183, -0.0007414, 0.0007165
5: 0.0058708, 0.0124175, 0.0059841, 0.0125437, -0.0048177, 0.0046558
6: -0.0016109, 0.0000507, -0.0016429, 0.0000220, -0.0011817, 0.0012228
7: -0.0073054, -0.0030063, -0.0073883, -0.0030807, -0.0030574, 0.0031637
8: -0.0034060, -0.0011451, -0.0034496, -0.0011843, -0.0016078, 0.0016638
9: -0.0005360, 0.0020856, -0.0004907, 0.0021361, -0.0019292, 0.0018644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013771
time: 0.94 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013770
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9893550, 0.9932957, 0.9891959, 0.9933630, -0.0028264, 0.0029264
1: -0.0039164, -0.0029345, -0.0039561, -0.0029177, -0.0007043, 0.0007292
2: 0.0054973, 0.0107009, 0.0054085, 0.0109110, -0.0038643, 0.0037322
3: -0.0061437, -0.0037752, -0.0062393, -0.0037348, -0.0016987, 0.0017588
4: 0.0015919, 0.0025990, 0.0015747, 0.0026397, -0.0007479, 0.0007224
5: 0.0058736, 0.0124183, 0.0057619, 0.0126827, -0.0048602, 0.0046941
6: -0.0016111, 0.0000501, -0.0016782, 0.0000784, -0.0011914, 0.0012336
7: -0.0073060, -0.0030081, -0.0074795, -0.0029348, -0.0030826, 0.0031916
8: -0.0034063, -0.0011461, -0.0034976, -0.0011075, -0.0016211, 0.0016785
9: -0.0005349, 0.0020859, -0.0005796, 0.0021917, -0.0019462, 0.0018797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013839
time: 1.00 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013839
time: 0.90 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9893555, 0.9932973, 0.9891996, 0.9933331, -0.0028659, 0.0029056
1: -0.0039163, -0.0029341, -0.0039551, -0.0029252, -0.0007141, 0.0007240
2: 0.0054951, 0.0107002, 0.0054480, 0.0109061, -0.0038368, 0.0037844
3: -0.0061434, -0.0037743, -0.0062371, -0.0037528, -0.0017225, 0.0017463
4: 0.0015915, 0.0025989, 0.0015823, 0.0026387, -0.0007426, 0.0007325
5: 0.0058708, 0.0124175, 0.0058115, 0.0126765, -0.0048256, 0.0047598
6: -0.0016109, 0.0000507, -0.0016766, 0.0000658, -0.0012081, 0.0012248
7: -0.0073054, -0.0030063, -0.0074755, -0.0029674, -0.0031257, 0.0031689
8: -0.0034060, -0.0011451, -0.0034954, -0.0011247, -0.0016438, 0.0016665
9: -0.0005360, 0.0020856, -0.0005598, 0.0021893, -0.0019324, 0.0019060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013770
time: 0.92 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013770
time: 0.99 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.54 seconds
IS_A1_A1_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013279
IS_A1_A1_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013279
IS_A1_A1_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013279
IS_A1_A1_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013279
IS_A1_A1_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013730
IS_A1_A1_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013731
IS_A1_A1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013731
IS_A1_A1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0013731
IS_A1_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013205
IS_A1_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013205
IS_A1_A1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013205
IS_A1_A1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014071, upper bound: 0.0013205
IS_A1_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013654
IS_A1_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013654
IS_A1_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013654
IS_A1_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014072, upper bound: 0.0013654
IS_A1_A1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013597, upper bound: 0.0013079
IS_A1_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013682, upper bound: 0.0013051
IS_A1_A1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013597, upper bound: 0.0013079
IS_A1_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013682, upper bound: 0.0013051
IS_A1_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013676, upper bound: 0.0013056
IS_A1_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013028
IS_A1_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013676, upper bound: 0.0013056
IS_A1_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013768, upper bound: 0.0013028
IS_A1_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013594, upper bound: 0.0013766
IS_A1_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013661
IS_A1_A1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013594, upper bound: 0.0013766
IS_A1_A1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013661
IS_A1_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0013444
IS_A1_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0013448
IS_A1_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0013444
IS_A1_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0013448
IS_A1_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013211, upper bound: 0.0013386
IS_A1_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013120, upper bound: 0.0013404
IS_A1_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013211, upper bound: 0.0013386
IS_A1_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013120, upper bound: 0.0013403
IS_A1_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013674, upper bound: 0.0013385
IS_A1_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013569, upper bound: 0.0013404
IS_A1_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013674, upper bound: 0.0013385
IS_A1_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013569, upper bound: 0.0013403
IS_A1_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0013536
IS_A1_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013032, upper bound: 0.0013555
IS_A1_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0013536
IS_A1_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013032, upper bound: 0.0013555
IS_A1_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013436, upper bound: 0.0013605
IS_A1_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013478, upper bound: 0.0013555
IS_A1_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013575, upper bound: 0.0013536
IS_A1_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013478, upper bound: 0.0013555
IS_A1_A2_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013250, upper bound: 0.0012990
IS_A1_A2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0012990
IS_A1_A2_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013355, upper bound: 0.0012909
IS_A1_A2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0012909
IS_A1_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013793, upper bound: 0.0012718
IS_A1_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0012718
IS_A1_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013727, upper bound: 0.0012735
IS_A1_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0012735
IS_A1_A2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013250, upper bound: 0.0013483
IS_A1_A2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0013482
IS_A1_A2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013355, upper bound: 0.0013386
IS_A1_A2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0013386
IS_A1_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013096
IS_A1_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013138
IS_A1_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013771, upper bound: 0.0013133
IS_A1_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0013167
IS_A2_A1_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
IS_A2_A1_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
IS_A2_A1_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
IS_A2_A1_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
IS_A2_A1_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
IS_A2_A1_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
IS_A2_A1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
IS_A2_A1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
IS_A2_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013808
IS_A2_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013809
IS_A2_A1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013808
IS_A2_A1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013809
IS_A2_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
IS_A2_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
IS_A2_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014223
IS_A2_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
IS_A2_A1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013045, upper bound: 0.0014206
IS_A2_A1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013050, upper bound: 0.0014091
IS_A2_A1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0013918
IS_A2_A1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013050, upper bound: 0.0014091
IS_A2_A1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013140, upper bound: 0.0014166
IS_A2_A1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013143, upper bound: 0.0014068
IS_A2_A1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013189, upper bound: 0.0013901
IS_A2_A1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013143, upper bound: 0.0014068
IS_A2_A1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013466, upper bound: 0.0014206
IS_A2_A1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013475, upper bound: 0.0014091
IS_A2_A1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013525, upper bound: 0.0013918
IS_A2_A1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013475, upper bound: 0.0014091
IS_A2_A1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014166
IS_A2_A1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014068
IS_A2_A1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013605, upper bound: 0.0013901
IS_A2_A1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014068
IS_A2_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013010, upper bound: 0.0013788
IS_A2_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0012990, upper bound: 0.0013940
IS_A2_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013010, upper bound: 0.0013787
IS_A2_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0012990, upper bound: 0.0013940
IS_A2_A2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014067
IS_A2_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013403, upper bound: 0.0013939
IS_A2_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013366, upper bound: 0.0014011
IS_A2_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013430, upper bound: 0.0014011
IS_A2_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014299
IS_A2_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014185
IS_A2_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0012925, upper bound: 0.0013971
IS_A2_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014185
IS_A2_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013306, upper bound: 0.0014299
IS_A2_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013318, upper bound: 0.0014185
IS_A2_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013253, upper bound: 0.0014178
IS_A2_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013304, upper bound: 0.0014178
IS_A2_A2_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013381
IS_A2_A2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013381
IS_A2_A2_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013294
IS_A2_A2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013295
IS_A2_A2_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013381
IS_A2_A2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013381
IS_A2_A2_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013295
IS_A2_A2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013295
IS_A2_A2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013839
IS_A2_A2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013839
IS_A2_A2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013771
IS_A2_A2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013770
IS_A2_A2_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013839
IS_A2_A2_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013839
IS_A2_A2_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013770
IS_A2_A2_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013770

## BFS IS instance: IS_A1_A1_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9895820, 0.9932273, -0.0025084, 0.0024576
1: -0.0038766, -0.0029508, -0.0038598, -0.0029515, -0.0006250, 0.0006124
2: 0.0055839, 0.0104897, 0.0055876, 0.0104011, -0.0032452, 0.0033123
3: -0.0060476, -0.0038147, -0.0060073, -0.0038163, -0.0015076, 0.0014771
4: 0.0016086, 0.0025582, 0.0016094, 0.0025410, -0.0006281, 0.0006411
5: 0.0059825, 0.0121528, 0.0059872, 0.0120413, -0.0040816, 0.0041660
6: -0.0015437, 0.0000224, -0.0015154, 0.0000212, -0.0010574, 0.0010360
7: -0.0071316, -0.0030797, -0.0070584, -0.0030827, -0.0027358, 0.0026803
8: -0.0033146, -0.0011837, -0.0032761, -0.0011853, -0.0014387, 0.0014096
9: -0.0004913, 0.0019796, -0.0004894, 0.0019349, -0.0016345, 0.0016683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013577, upper bound: 0.0013046
time: 0.86 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013613, upper bound: 0.0013026
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9895051, 0.9933148, -0.0025582, 0.0024939
1: -0.0038766, -0.0029508, -0.0038790, -0.0029297, -0.0006374, 0.0006214
2: 0.0055839, 0.0104897, 0.0054720, 0.0105027, -0.0032932, 0.0033780
3: -0.0060476, -0.0038147, -0.0060535, -0.0037638, -0.0015375, 0.0014989
4: 0.0016086, 0.0025582, 0.0015870, 0.0025607, -0.0006374, 0.0006538
5: 0.0059825, 0.0121528, 0.0058418, 0.0121691, -0.0041420, 0.0042487
6: -0.0015437, 0.0000224, -0.0015478, 0.0000581, -0.0010784, 0.0010513
7: -0.0071316, -0.0030797, -0.0071423, -0.0029873, -0.0027901, 0.0027200
8: -0.0033146, -0.0011837, -0.0033202, -0.0011351, -0.0014673, 0.0014304
9: -0.0004913, 0.0019796, -0.0005476, 0.0019861, -0.0016586, 0.0017014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013577, upper bound: 0.0013046
time: 0.83 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013613, upper bound: 0.0013026
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9894842, 0.9932004, -0.0024603, 0.0025018
1: -0.0038766, -0.0029508, -0.0038842, -0.0029582, -0.0006130, 0.0006234
2: 0.0055839, 0.0104897, 0.0056230, 0.0105302, -0.0033036, 0.0032488
3: -0.0060476, -0.0038147, -0.0060660, -0.0038325, -0.0014787, 0.0015037
4: 0.0016086, 0.0025582, 0.0016162, 0.0025660, -0.0006394, 0.0006288
5: 0.0059825, 0.0121528, 0.0060317, 0.0122036, -0.0041551, 0.0040862
6: -0.0015437, 0.0000224, -0.0015566, 0.0000099, -0.0010371, 0.0010546
7: -0.0071316, -0.0030797, -0.0071650, -0.0031119, -0.0026833, 0.0027286
8: -0.0033146, -0.0011837, -0.0033321, -0.0012007, -0.0014111, 0.0014349
9: -0.0004913, 0.0019796, -0.0004716, 0.0019999, -0.0016639, 0.0016363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013518, upper bound: 0.0012962
time: 0.82 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013569, upper bound: 0.0012943
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9894134, 0.9932876, -0.0025071, 0.0025910
1: -0.0038766, -0.0029508, -0.0039018, -0.0029365, -0.0006247, 0.0006456
2: 0.0055839, 0.0104897, 0.0055080, 0.0106237, -0.0034214, 0.0033106
3: -0.0060476, -0.0038147, -0.0061086, -0.0037801, -0.0015069, 0.0015573
4: 0.0016086, 0.0025582, 0.0015939, 0.0025841, -0.0006622, 0.0006408
5: 0.0059825, 0.0121528, 0.0058871, 0.0123213, -0.0043032, 0.0041639
6: -0.0015437, 0.0000224, -0.0015864, 0.0000466, -0.0010568, 0.0010922
7: -0.0071316, -0.0030797, -0.0072422, -0.0030170, -0.0027344, 0.0028259
8: -0.0033146, -0.0011837, -0.0033728, -0.0011507, -0.0014380, 0.0014861
9: -0.0004913, 0.0019796, -0.0005295, 0.0020470, -0.0017232, 0.0016674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013518, upper bound: 0.0012962
time: 0.81 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013569, upper bound: 0.0012943
time: 0.99 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9895385, 0.9933221, -0.0025400, 0.0025011
1: -0.0038792, -0.0029507, -0.0038707, -0.0029279, -0.0006329, 0.0006232
2: 0.0055833, 0.0105038, 0.0054624, 0.0104586, -0.0033027, 0.0033541
3: -0.0060540, -0.0038144, -0.0060334, -0.0037594, -0.0015266, 0.0015033
4: 0.0016085, 0.0025609, 0.0015851, 0.0025521, -0.0006392, 0.0006492
5: 0.0059818, 0.0121705, 0.0058298, 0.0121136, -0.0041540, 0.0042185
6: -0.0015482, 0.0000226, -0.0015337, 0.0000612, -0.0010707, 0.0010543
7: -0.0071432, -0.0030792, -0.0071058, -0.0029793, -0.0027703, 0.0027279
8: -0.0033207, -0.0011835, -0.0033010, -0.0011310, -0.0014569, 0.0014346
9: -0.0004916, 0.0019867, -0.0005525, 0.0019639, -0.0016634, 0.0016893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0013635
time: 0.87 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013532, upper bound: 0.0013532
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9894425, 0.9932938, -0.0024965, 0.0026089
1: -0.0038792, -0.0029507, -0.0038946, -0.0029350, -0.0006221, 0.0006501
2: 0.0055833, 0.0105038, 0.0054998, 0.0105854, -0.0034450, 0.0032966
3: -0.0060540, -0.0038144, -0.0060911, -0.0037764, -0.0015005, 0.0015680
4: 0.0016085, 0.0025609, 0.0015924, 0.0025767, -0.0006668, 0.0006381
5: 0.0059818, 0.0121705, 0.0058767, 0.0122731, -0.0043329, 0.0041463
6: -0.0015482, 0.0000226, -0.0015742, 0.0000493, -0.0010524, 0.0010997
7: -0.0071432, -0.0030792, -0.0072106, -0.0030102, -0.0027228, 0.0028454
8: -0.0033207, -0.0011835, -0.0033561, -0.0011472, -0.0014319, 0.0014963
9: -0.0004916, 0.0019867, -0.0005337, 0.0020277, -0.0017351, 0.0016604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0013635
time: 0.88 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013532, upper bound: 0.0013531
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9894607, 0.9934159, -0.0026068, 0.0025394
1: -0.0038792, -0.0029507, -0.0038900, -0.0029045, -0.0006495, 0.0006327
2: 0.0055833, 0.0105038, 0.0053385, 0.0105612, -0.0033532, 0.0034423
3: -0.0060540, -0.0038144, -0.0060801, -0.0037030, -0.0015668, 0.0015262
4: 0.0016085, 0.0025609, 0.0015611, 0.0025720, -0.0006490, 0.0006662
5: 0.0059818, 0.0121705, 0.0056739, 0.0122427, -0.0042175, 0.0043295
6: -0.0015482, 0.0000226, -0.0015665, 0.0001007, -0.0010989, 0.0010704
7: -0.0071432, -0.0030792, -0.0071906, -0.0028770, -0.0028431, 0.0027696
8: -0.0033207, -0.0011835, -0.0033456, -0.0010771, -0.0014952, 0.0014565
9: -0.0004916, 0.0019867, -0.0006149, 0.0020156, -0.0016889, 0.0017337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014033, upper bound: 0.0013381
time: 0.87 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013918, upper bound: 0.0013384
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9893707, 0.9933856, -0.0025602, 0.0026337
1: -0.0038792, -0.0029507, -0.0039125, -0.0029121, -0.0006379, 0.0006562
2: 0.0055833, 0.0105038, 0.0053786, 0.0106801, -0.0034777, 0.0033807
3: -0.0060540, -0.0038144, -0.0061342, -0.0037212, -0.0015388, 0.0015829
4: 0.0016085, 0.0025609, 0.0015689, 0.0025950, -0.0006731, 0.0006543
5: 0.0059818, 0.0121705, 0.0057243, 0.0123922, -0.0043740, 0.0042520
6: -0.0015482, 0.0000226, -0.0016044, 0.0000880, -0.0010792, 0.0011102
7: -0.0071432, -0.0030792, -0.0072888, -0.0029101, -0.0027923, 0.0028724
8: -0.0033207, -0.0011835, -0.0033973, -0.0010945, -0.0014684, 0.0015106
9: -0.0004916, 0.0019867, -0.0005947, 0.0020754, -0.0017516, 0.0017027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014033, upper bound: 0.0013381
time: 1.07 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013918, upper bound: 0.0013384
time: 0.80 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9892556, 0.9931098, -0.0024131, 0.0028303
1: -0.0038766, -0.0029508, -0.0039412, -0.0029808, -0.0006013, 0.0007052
2: 0.0055839, 0.0104897, 0.0057427, 0.0108322, -0.0037374, 0.0031865
3: -0.0060476, -0.0038147, -0.0062035, -0.0038870, -0.0014504, 0.0017011
4: 0.0016086, 0.0025582, 0.0016394, 0.0026244, -0.0007234, 0.0006167
5: 0.0059825, 0.0121528, 0.0061823, 0.0125835, -0.0047007, 0.0040078
6: -0.0015437, 0.0000224, -0.0016530, -0.0000283, -0.0010172, 0.0011931
7: -0.0071316, -0.0030797, -0.0074145, -0.0032108, -0.0026319, 0.0030869
8: -0.0033146, -0.0011837, -0.0034633, -0.0012527, -0.0013841, 0.0016234
9: -0.0004913, 0.0019796, -0.0004113, 0.0021521, -0.0018824, 0.0016049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013593, upper bound: 0.0013107
time: 0.82 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013028
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9891263, 0.9931122, -0.0024276, 0.0029650
1: -0.0038766, -0.0029508, -0.0039734, -0.0029802, -0.0006049, 0.0007388
2: 0.0055839, 0.0104897, 0.0057396, 0.0110029, -0.0039153, 0.0032056
3: -0.0060476, -0.0038147, -0.0062812, -0.0038855, -0.0014591, 0.0017821
4: 0.0016086, 0.0025582, 0.0016388, 0.0026575, -0.0007578, 0.0006204
5: 0.0059825, 0.0121528, 0.0061784, 0.0127982, -0.0049244, 0.0040319
6: -0.0015437, 0.0000224, -0.0017075, -0.0000273, -0.0010233, 0.0012499
7: -0.0071316, -0.0030797, -0.0075554, -0.0032083, -0.0026477, 0.0032338
8: -0.0033146, -0.0011837, -0.0035375, -0.0012513, -0.0013924, 0.0017006
9: -0.0004913, 0.0019796, -0.0004129, 0.0022380, -0.0019720, 0.0016145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013593, upper bound: 0.0013107
time: 0.82 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013028
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9891739, 0.9932092, -0.0024592, 0.0028606
1: -0.0038766, -0.0029508, -0.0039615, -0.0029560, -0.0006128, 0.0007128
2: 0.0055839, 0.0104897, 0.0056115, 0.0109400, -0.0037774, 0.0032473
3: -0.0060476, -0.0038147, -0.0062525, -0.0038272, -0.0014780, 0.0017193
4: 0.0016086, 0.0025582, 0.0016140, 0.0026453, -0.0007311, 0.0006285
5: 0.0059825, 0.0121528, 0.0060172, 0.0127190, -0.0047510, 0.0040843
6: -0.0015437, 0.0000224, -0.0016874, 0.0000136, -0.0010366, 0.0012058
7: -0.0071316, -0.0030797, -0.0075034, -0.0031024, -0.0026821, 0.0031199
8: -0.0033146, -0.0011837, -0.0035101, -0.0011957, -0.0014105, 0.0016407
9: -0.0004913, 0.0019796, -0.0004774, 0.0022063, -0.0019025, 0.0016355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0012870
time: 0.81 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0012870
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9895149, 0.9932301, 0.9890425, 0.9932153, -0.0024706, 0.0030382
1: -0.0038766, -0.0029508, -0.0039943, -0.0029545, -0.0006156, 0.0007570
2: 0.0055839, 0.0104897, 0.0056035, 0.0111135, -0.0040120, 0.0032624
3: -0.0060476, -0.0038147, -0.0063315, -0.0038236, -0.0014849, 0.0018261
4: 0.0016086, 0.0025582, 0.0016124, 0.0026789, -0.0007765, 0.0006314
5: 0.0059825, 0.0121528, 0.0060072, 0.0129373, -0.0050460, 0.0041033
6: -0.0015437, 0.0000224, -0.0017428, 0.0000161, -0.0010415, 0.0012807
7: -0.0071316, -0.0030797, -0.0076468, -0.0030959, -0.0026946, 0.0033136
8: -0.0033146, -0.0011837, -0.0035855, -0.0011922, -0.0014170, 0.0017426
9: -0.0004913, 0.0019796, -0.0004814, 0.0022937, -0.0020206, 0.0016431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0012870
time: 0.81 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0012870
time: 1.19 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9892120, 0.9932087, -0.0024616, 0.0028691
1: -0.0038792, -0.0029507, -0.0039520, -0.0029562, -0.0006134, 0.0007149
2: 0.0055833, 0.0105038, 0.0056121, 0.0108897, -0.0037886, 0.0032505
3: -0.0060540, -0.0038144, -0.0062296, -0.0038275, -0.0014795, 0.0017244
4: 0.0016085, 0.0025609, 0.0016141, 0.0026356, -0.0007333, 0.0006291
5: 0.0059818, 0.0121705, 0.0060180, 0.0126558, -0.0047651, 0.0040883
6: -0.0015482, 0.0000226, -0.0016714, 0.0000134, -0.0010376, 0.0012094
7: -0.0071432, -0.0030792, -0.0074619, -0.0031030, -0.0026847, 0.0031292
8: -0.0033207, -0.0011835, -0.0034883, -0.0011960, -0.0014119, 0.0016456
9: -0.0004916, 0.0019867, -0.0004771, 0.0021810, -0.0019082, 0.0016371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013792, upper bound: 0.0013417
time: 0.90 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013445
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9890873, 0.9932092, -0.0024778, 0.0030607
1: -0.0038792, -0.0029507, -0.0039831, -0.0029560, -0.0006174, 0.0007626
2: 0.0055833, 0.0105038, 0.0056114, 0.0110544, -0.0040416, 0.0032719
3: -0.0060540, -0.0038144, -0.0063046, -0.0038272, -0.0014892, 0.0018396
4: 0.0016085, 0.0025609, 0.0016140, 0.0026675, -0.0007822, 0.0006333
5: 0.0059818, 0.0121705, 0.0060171, 0.0128630, -0.0050833, 0.0041152
6: -0.0015482, 0.0000226, -0.0017239, 0.0000136, -0.0010445, 0.0012902
7: -0.0071432, -0.0030792, -0.0075980, -0.0031024, -0.0027024, 0.0033381
8: -0.0033207, -0.0011835, -0.0035599, -0.0011957, -0.0014212, 0.0017555
9: -0.0004916, 0.0019867, -0.0004774, 0.0022640, -0.0020356, 0.0016479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013792, upper bound: 0.0013417
time: 1.09 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013445
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9891304, 0.9933133, -0.0025260, 0.0029014
1: -0.0038792, -0.0029507, -0.0039724, -0.0029301, -0.0006294, 0.0007230
2: 0.0055833, 0.0105038, 0.0054740, 0.0109975, -0.0038313, 0.0033356
3: -0.0060540, -0.0038144, -0.0062787, -0.0037646, -0.0015182, 0.0017438
4: 0.0016085, 0.0025609, 0.0015874, 0.0026564, -0.0007415, 0.0006456
5: 0.0059818, 0.0121705, 0.0058443, 0.0127914, -0.0048188, 0.0041953
6: -0.0015482, 0.0000226, -0.0017058, 0.0000575, -0.0010648, 0.0012231
7: -0.0071432, -0.0030792, -0.0075509, -0.0029889, -0.0027550, 0.0031644
8: -0.0033207, -0.0011835, -0.0035351, -0.0011360, -0.0014488, 0.0016641
9: -0.0004916, 0.0019867, -0.0005466, 0.0022353, -0.0019296, 0.0016800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0013304
time: 0.82 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0013306
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9895042, 0.9932305, 0.9890040, 0.9933147, -0.0025313, 0.0030766
1: -0.0038792, -0.0029507, -0.0040039, -0.0029298, -0.0006307, 0.0007666
2: 0.0055833, 0.0105038, 0.0054722, 0.0111644, -0.0040626, 0.0033426
3: -0.0060540, -0.0038144, -0.0063547, -0.0037638, -0.0015214, 0.0018491
4: 0.0016085, 0.0025609, 0.0015870, 0.0026887, -0.0007863, 0.0006469
5: 0.0059818, 0.0121705, 0.0058420, 0.0130013, -0.0051097, 0.0042041
6: -0.0015482, 0.0000226, -0.0017590, 0.0000581, -0.0010670, 0.0012969
7: -0.0071432, -0.0030792, -0.0076888, -0.0029874, -0.0027608, 0.0033555
8: -0.0033207, -0.0011835, -0.0036076, -0.0011352, -0.0014519, 0.0017646
9: -0.0004916, 0.0019867, -0.0005476, 0.0023194, -0.0020462, 0.0016835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0013304
time: 1.01 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0013306
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9892100, 0.9931097, 0.9896542, 0.9932294, -0.0028287, 0.0024348
1: -0.0039525, -0.0029808, -0.0038418, -0.0029510, -0.0007048, 0.0006067
2: 0.0057428, 0.0108923, 0.0055849, 0.0103058, -0.0032151, 0.0037353
3: -0.0062308, -0.0038870, -0.0059639, -0.0038151, -0.0017001, 0.0014634
4: 0.0016394, 0.0026361, 0.0016088, 0.0025225, -0.0006223, 0.0007230
5: 0.0061824, 0.0126591, 0.0059838, 0.0119214, -0.0040437, 0.0046980
6: -0.0016722, -0.0000283, -0.0014849, 0.0000221, -0.0011924, 0.0010263
7: -0.0074640, -0.0032109, -0.0069796, -0.0030805, -0.0030851, 0.0026554
8: -0.0034894, -0.0012527, -0.0032347, -0.0011841, -0.0016224, 0.0013965
9: -0.0004112, 0.0021823, -0.0004908, 0.0018869, -0.0016193, 0.0018813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013419, upper bound: 0.0013340
time: 0.84 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013419, upper bound: 0.0013340
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9892087, 0.9931108, 0.9896537, 0.9932074, -0.0028569, 0.0024697
1: -0.0039529, -0.0029805, -0.0038420, -0.0029565, -0.0007119, 0.0006154
2: 0.0057414, 0.0108941, 0.0056139, 0.0103066, -0.0032612, 0.0037725
3: -0.0062317, -0.0038863, -0.0059642, -0.0038283, -0.0017171, 0.0014844
4: 0.0016391, 0.0026364, 0.0016144, 0.0025227, -0.0006312, 0.0007302
5: 0.0061806, 0.0126614, 0.0060202, 0.0119224, -0.0041017, 0.0047448
6: -0.0016728, -0.0000279, -0.0014852, 0.0000128, -0.0012043, 0.0010411
7: -0.0074656, -0.0032097, -0.0069803, -0.0031044, -0.0031158, 0.0026935
8: -0.0034902, -0.0012521, -0.0032350, -0.0011967, -0.0016386, 0.0014165
9: -0.0004120, 0.0021832, -0.0004762, 0.0018873, -0.0016425, 0.0019000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0013244
time: 0.85 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0013244
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9892100, 0.9931097, 0.9895797, 0.9933234, -0.0028914, 0.0025326
1: -0.0039525, -0.0029808, -0.0038604, -0.0029276, -0.0007205, 0.0006311
2: 0.0057428, 0.0108923, 0.0054606, 0.0104041, -0.0033443, 0.0038181
3: -0.0062308, -0.0038870, -0.0060086, -0.0037586, -0.0017378, 0.0015222
4: 0.0016394, 0.0026361, 0.0015848, 0.0025416, -0.0006473, 0.0007390
5: 0.0061824, 0.0126591, 0.0058275, 0.0120451, -0.0042062, 0.0048022
6: -0.0016722, -0.0000283, -0.0015163, 0.0000618, -0.0012188, 0.0010676
7: -0.0074640, -0.0032109, -0.0070609, -0.0029778, -0.0031535, 0.0027622
8: -0.0034894, -0.0012527, -0.0032774, -0.0011302, -0.0016584, 0.0014526
9: -0.0004112, 0.0021823, -0.0005534, 0.0019364, -0.0016844, 0.0019230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013709, upper bound: 0.0013079
time: 0.93 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013709, upper bound: 0.0013079
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9892087, 0.9931108, 0.9895806, 0.9933009, -0.0029034, 0.0025115
1: -0.0039529, -0.0029805, -0.0038602, -0.0029332, -0.0007235, 0.0006258
2: 0.0057414, 0.0108941, 0.0054903, 0.0104030, -0.0033165, 0.0038340
3: -0.0062317, -0.0038863, -0.0060081, -0.0037721, -0.0017451, 0.0015095
4: 0.0016391, 0.0026364, 0.0015905, 0.0025414, -0.0006419, 0.0007421
5: 0.0061806, 0.0126614, 0.0058648, 0.0120436, -0.0041712, 0.0048221
6: -0.0016728, -0.0000279, -0.0015160, 0.0000523, -0.0012239, 0.0010587
7: -0.0074656, -0.0032097, -0.0070599, -0.0030023, -0.0031666, 0.0027392
8: -0.0034902, -0.0012521, -0.0032769, -0.0011430, -0.0016653, 0.0014405
9: -0.0004120, 0.0021832, -0.0005384, 0.0019359, -0.0016703, 0.0019310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013843, upper bound: 0.0013051
time: 0.82 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013843, upper bound: 0.0013051
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9892100, 0.9931097, 0.9895529, 0.9932154, -0.0028240, 0.0025439
1: -0.0039525, -0.0029808, -0.0038671, -0.0029545, -0.0007037, 0.0006339
2: 0.0057428, 0.0108923, 0.0056033, 0.0104395, -0.0033591, 0.0037290
3: -0.0062308, -0.0038870, -0.0060247, -0.0038235, -0.0016973, 0.0015289
4: 0.0016394, 0.0026361, 0.0016124, 0.0025484, -0.0006502, 0.0007217
5: 0.0061824, 0.0126591, 0.0060069, 0.0120895, -0.0042249, 0.0046901
6: -0.0016722, -0.0000283, -0.0015276, 0.0000162, -0.0011904, 0.0010723
7: -0.0074640, -0.0032109, -0.0070900, -0.0030957, -0.0030799, 0.0027744
8: -0.0034894, -0.0012527, -0.0032927, -0.0011921, -0.0016197, 0.0014590
9: -0.0004112, 0.0021823, -0.0004815, 0.0019542, -0.0016918, 0.0018781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013436, upper bound: 0.0013313
time: 0.92 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013436, upper bound: 0.0013313
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9892087, 0.9931108, 0.9895563, 0.9931890, -0.0028525, 0.0024954
1: -0.0039529, -0.0029805, -0.0038662, -0.0029611, -0.0007108, 0.0006218
2: 0.0057414, 0.0108941, 0.0056382, 0.0104350, -0.0032951, 0.0037667
3: -0.0062317, -0.0038863, -0.0060227, -0.0038394, -0.0017145, 0.0014998
4: 0.0016391, 0.0026364, 0.0016191, 0.0025476, -0.0006378, 0.0007290
5: 0.0061806, 0.0126614, 0.0060508, 0.0120840, -0.0041444, 0.0047376
6: -0.0016728, -0.0000279, -0.0015262, 0.0000051, -0.0012024, 0.0010519
7: -0.0074656, -0.0032097, -0.0070864, -0.0031245, -0.0031111, 0.0027215
8: -0.0034902, -0.0012521, -0.0032908, -0.0012073, -0.0016361, 0.0014312
9: -0.0004120, 0.0021832, -0.0004639, 0.0019520, -0.0016596, 0.0018971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013478, upper bound: 0.0013222
time: 0.91 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013478, upper bound: 0.0013222
time: 0.94 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.9892100, 0.9931097, 0.9894866, 0.9933117, -0.0028860, 0.0025796
1: -0.0039525, -0.0029808, -0.0038836, -0.0029305, -0.0007191, 0.0006428
2: 0.0057428, 0.0108923, 0.0054762, 0.0105272, -0.0034064, 0.0038109
3: -0.0062308, -0.0038870, -0.0060646, -0.0037656, -0.0017346, 0.0015504
4: 0.0016394, 0.0026361, 0.0015878, 0.0025654, -0.0006593, 0.0007376
5: 0.0061824, 0.0126591, 0.0058470, 0.0121998, -0.0042843, 0.0047931
6: -0.0016722, -0.0000283, -0.0015556, 0.0000568, -0.0012165, 0.0010874
7: -0.0074640, -0.0032109, -0.0071625, -0.0029907, -0.0031476, 0.0028135
8: -0.0034894, -0.0012527, -0.0033308, -0.0011369, -0.0016553, 0.0014796
9: -0.0004112, 0.0021823, -0.0005455, 0.0019984, -0.0017156, 0.0019194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013543, upper bound: 0.0012656
time: 0.95 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013544, upper bound: 0.0012635
time: 1.00 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.9892087, 0.9931108, 0.9894891, 0.9932846, -0.0028929, 0.0026063
1: -0.0039529, -0.0029805, -0.0038830, -0.0029373, -0.0007208, 0.0006494
2: 0.0057414, 0.0108941, 0.0055119, 0.0105238, -0.0034416, 0.0038201
3: -0.0062317, -0.0038863, -0.0060631, -0.0037819, -0.0017387, 0.0015665
4: 0.0016391, 0.0026364, 0.0015947, 0.0025647, -0.0006661, 0.0007394
5: 0.0061806, 0.0126614, 0.0058920, 0.0121956, -0.0043286, 0.0048047
6: -0.0016728, -0.0000279, -0.0015545, 0.0000454, -0.0012195, 0.0010986
7: -0.0074656, -0.0032097, -0.0071597, -0.0030202, -0.0031551, 0.0028425
8: -0.0034902, -0.0012521, -0.0033294, -0.0011525, -0.0016593, 0.0014949
9: -0.0004120, 0.0021832, -0.0005275, 0.0019967, -0.0017334, 0.0019240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013771, upper bound: 0.0012635
time: 0.91 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0012618
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.9891993, 0.9931103, 0.9896105, 0.9933279, -0.0028728, 0.0025351
1: -0.0039552, -0.0029807, -0.0038527, -0.0029265, -0.0007158, 0.0006317
2: 0.0057420, 0.0109065, 0.0054547, 0.0103635, -0.0033476, 0.0037935
3: -0.0062373, -0.0038866, -0.0059901, -0.0037559, -0.0017267, 0.0015237
4: 0.0016392, 0.0026388, 0.0015836, 0.0025337, -0.0006479, 0.0007342
5: 0.0061814, 0.0126769, 0.0058200, 0.0119939, -0.0042104, 0.0047713
6: -0.0016767, -0.0000281, -0.0015034, 0.0000637, -0.0012110, 0.0010687
7: -0.0074758, -0.0032102, -0.0070273, -0.0029729, -0.0031332, 0.0027649
8: -0.0034956, -0.0012524, -0.0032597, -0.0011276, -0.0016477, 0.0014541
9: -0.0004117, 0.0021895, -0.0005564, 0.0019160, -0.0016860, 0.0019106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013419, upper bound: 0.0013798
time: 0.86 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013419, upper bound: 0.0013798
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9891978, 0.9931115, 0.9896114, 0.9933075, -0.0028999, 0.0025167
1: -0.0039556, -0.0029804, -0.0038525, -0.0029315, -0.0007226, 0.0006271
2: 0.0057405, 0.0109085, 0.0054817, 0.0103623, -0.0033232, 0.0038293
3: -0.0062382, -0.0038859, -0.0059896, -0.0037681, -0.0017429, 0.0015126
4: 0.0016389, 0.0026392, 0.0015889, 0.0025335, -0.0006432, 0.0007412
5: 0.0061795, 0.0126795, 0.0058540, 0.0119925, -0.0041798, 0.0048163
6: -0.0016774, -0.0000276, -0.0015030, 0.0000550, -0.0012224, 0.0010609
7: -0.0074775, -0.0032090, -0.0070263, -0.0029952, -0.0031628, 0.0027448
8: -0.0034965, -0.0012517, -0.0032592, -0.0011393, -0.0016633, 0.0014435
9: -0.0004124, 0.0021905, -0.0005428, 0.0019154, -0.0016738, 0.0019286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0013682
time: 0.84 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0013682
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.9891993, 0.9931103, 0.9895157, 0.9933107, -0.0028735, 0.0026434
1: -0.0039552, -0.0029807, -0.0038764, -0.0029307, -0.0007160, 0.0006587
2: 0.0057420, 0.0109065, 0.0054773, 0.0104888, -0.0034905, 0.0037944
3: -0.0062373, -0.0038866, -0.0060471, -0.0037662, -0.0017270, 0.0015887
4: 0.0016392, 0.0026388, 0.0015880, 0.0025580, -0.0006756, 0.0007344
5: 0.0061814, 0.0126769, 0.0058485, 0.0121515, -0.0043902, 0.0047723
6: -0.0016767, -0.0000281, -0.0015434, 0.0000564, -0.0012113, 0.0011143
7: -0.0074758, -0.0032102, -0.0071308, -0.0029916, -0.0031339, 0.0028830
8: -0.0034956, -0.0012524, -0.0033141, -0.0011374, -0.0016481, 0.0015161
9: -0.0004117, 0.0021895, -0.0005450, 0.0019791, -0.0017580, 0.0019111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0013309
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0013362
time: 0.98 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.9891978, 0.9931115, 0.9895166, 0.9932852, -0.0028989, 0.0026223
1: -0.0039556, -0.0029804, -0.0038761, -0.0029371, -0.0007223, 0.0006534
2: 0.0057405, 0.0109085, 0.0055112, 0.0104875, -0.0034627, 0.0038279
3: -0.0062382, -0.0038859, -0.0060466, -0.0037816, -0.0017423, 0.0015761
4: 0.0016389, 0.0026392, 0.0015946, 0.0025577, -0.0006702, 0.0007409
5: 0.0061795, 0.0126795, 0.0058911, 0.0121499, -0.0043551, 0.0048146
6: -0.0016774, -0.0000276, -0.0015430, 0.0000456, -0.0012220, 0.0011054
7: -0.0074775, -0.0032090, -0.0071297, -0.0030196, -0.0031616, 0.0028599
8: -0.0034965, -0.0012517, -0.0033136, -0.0011521, -0.0016627, 0.0015040
9: -0.0004124, 0.0021905, -0.0005279, 0.0019784, -0.0017440, 0.0019280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0013227
time: 0.90 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0013269
time: 1.01 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9892485, 0.9931288, 0.9894834, 0.9934131, -0.0028977, 0.0026136
1: -0.0039429, -0.0029761, -0.0038844, -0.0029052, -0.0007220, 0.0006512
2: 0.0057177, 0.0108416, 0.0053422, 0.0105314, -0.0034512, 0.0038264
3: -0.0062077, -0.0038756, -0.0060665, -0.0037047, -0.0017416, 0.0015708
4: 0.0016345, 0.0026262, 0.0015619, 0.0025662, -0.0006680, 0.0007406
5: 0.0061508, 0.0125953, 0.0056785, 0.0122051, -0.0043407, 0.0048126
6: -0.0016560, -0.0000203, -0.0015570, 0.0000996, -0.0012215, 0.0011017
7: -0.0074222, -0.0031901, -0.0071660, -0.0028800, -0.0031604, 0.0028505
8: -0.0034674, -0.0012418, -0.0033327, -0.0010787, -0.0016620, 0.0014990
9: -0.0004239, 0.0021567, -0.0006130, 0.0020005, -0.0017382, 0.0019272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013757, upper bound: 0.0013015
time: 0.85 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013757, upper bound: 0.0013058
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9892497, 0.9931035, 0.9894825, 0.9934154, -0.0029014, 0.0026490
1: -0.0039426, -0.0029824, -0.0038846, -0.0029047, -0.0007230, 0.0006601
2: 0.0057510, 0.0108400, 0.0053392, 0.0105325, -0.0034979, 0.0038313
3: -0.0062070, -0.0038907, -0.0060671, -0.0037033, -0.0017439, 0.0015921
4: 0.0016410, 0.0026259, 0.0015613, 0.0025664, -0.0006770, 0.0007415
5: 0.0061927, 0.0125933, 0.0056747, 0.0122066, -0.0043995, 0.0048188
6: -0.0016555, -0.0000309, -0.0015573, 0.0001005, -0.0012231, 0.0011166
7: -0.0074208, -0.0032177, -0.0071669, -0.0028775, -0.0031644, 0.0028891
8: -0.0034667, -0.0012563, -0.0033332, -0.0010774, -0.0016641, 0.0015193
9: -0.0004071, 0.0021559, -0.0006145, 0.0020011, -0.0017617, 0.0019297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013675, upper bound: 0.0013037
time: 0.84 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013675, upper bound: 0.0013073
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9892485, 0.9931288, 0.9893923, 0.9933918, -0.0028829, 0.0027059
1: -0.0039429, -0.0029761, -0.0039071, -0.0029106, -0.0007183, 0.0006742
2: 0.0057177, 0.0108416, 0.0053704, 0.0106516, -0.0035731, 0.0038069
3: -0.0062077, -0.0038756, -0.0061212, -0.0037175, -0.0017327, 0.0016263
4: 0.0016345, 0.0026262, 0.0015673, 0.0025895, -0.0006916, 0.0007368
5: 0.0061508, 0.0125953, 0.0057140, 0.0123563, -0.0044940, 0.0047880
6: -0.0016560, -0.0000203, -0.0015953, 0.0000906, -0.0012153, 0.0011406
7: -0.0074222, -0.0031901, -0.0072652, -0.0029033, -0.0031442, 0.0029511
8: -0.0034674, -0.0012418, -0.0033849, -0.0010910, -0.0016535, 0.0015520
9: -0.0004239, 0.0021567, -0.0005988, 0.0020611, -0.0017996, 0.0019173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013007
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013040
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9892497, 0.9931035, 0.9893928, 0.9933933, -0.0028882, 0.0027366
1: -0.0039426, -0.0029824, -0.0039070, -0.0029102, -0.0007197, 0.0006819
2: 0.0057510, 0.0108400, 0.0053684, 0.0106509, -0.0036136, 0.0038138
3: -0.0062070, -0.0038907, -0.0061209, -0.0037166, -0.0017359, 0.0016448
4: 0.0016410, 0.0026259, 0.0015669, 0.0025893, -0.0006994, 0.0007382
5: 0.0061927, 0.0125933, 0.0057115, 0.0123555, -0.0045450, 0.0047968
6: -0.0016555, -0.0000309, -0.0015951, 0.0000912, -0.0012175, 0.0011536
7: -0.0074208, -0.0032177, -0.0072647, -0.0029017, -0.0031500, 0.0029846
8: -0.0034667, -0.0012563, -0.0033846, -0.0010901, -0.0016565, 0.0015696
9: -0.0004071, 0.0021559, -0.0005998, 0.0020607, -0.0018200, 0.0019208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0013020
time: 0.95 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013771, upper bound: 0.0013052
time: 1.01 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9895530, 0.9932111, 0.9895326, 0.9932464, -0.0024194, 0.0025402
1: -0.0038671, -0.0029556, -0.0038721, -0.0029468, -0.0006029, 0.0006329
2: 0.0056090, 0.0104394, 0.0055624, 0.0104663, -0.0033543, 0.0031948
3: -0.0060247, -0.0038261, -0.0060369, -0.0038049, -0.0014541, 0.0015267
4: 0.0016135, 0.0025484, 0.0016045, 0.0025536, -0.0006492, 0.0006184
5: 0.0060141, 0.0120894, 0.0059555, 0.0121233, -0.0042188, 0.0040183
6: -0.0015276, 0.0000144, -0.0015362, 0.0000293, -0.0010199, 0.0010708
7: -0.0070900, -0.0031004, -0.0071122, -0.0030619, -0.0026387, 0.0027704
8: -0.0032927, -0.0011946, -0.0033044, -0.0011744, -0.0013877, 0.0014569
9: -0.0004786, 0.0019542, -0.0005021, 0.0019678, -0.0016894, 0.0016091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012848, upper bound: 0.0013187
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B1_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012859, upper bound: 0.0013187
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9895564, 0.9931890, 0.9895310, 0.9932481, -0.0023843, 0.0025828
1: -0.0038662, -0.0029611, -0.0038726, -0.0029464, -0.0005941, 0.0006436
2: 0.0056382, 0.0104350, 0.0055601, 0.0104685, -0.0034106, 0.0031484
3: -0.0060227, -0.0038394, -0.0060379, -0.0038039, -0.0014330, 0.0015523
4: 0.0016191, 0.0025476, 0.0016040, 0.0025541, -0.0006601, 0.0006094
5: 0.0060508, 0.0120839, 0.0059527, 0.0121261, -0.0042896, 0.0039598
6: -0.0015262, 0.0000051, -0.0015369, 0.0000300, -0.0010050, 0.0010887
7: -0.0070864, -0.0031245, -0.0071141, -0.0030600, -0.0026004, 0.0028169
8: -0.0032908, -0.0012073, -0.0033054, -0.0011734, -0.0013675, 0.0014814
9: -0.0004640, 0.0019520, -0.0005032, 0.0019689, -0.0017177, 0.0015857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012767, upper bound: 0.0013242
time: 1.00 seconds

## Relational analysis of IS_A1_A2_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012772, upper bound: 0.0013242
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9895530, 0.9932111, 0.9894529, 0.9933312, -0.0024684, 0.0025875
1: -0.0038671, -0.0029556, -0.0038920, -0.0029257, -0.0006151, 0.0006447
2: 0.0056090, 0.0104394, 0.0054504, 0.0105716, -0.0034168, 0.0032595
3: -0.0060247, -0.0038261, -0.0060848, -0.0037539, -0.0014836, 0.0015552
4: 0.0016135, 0.0025484, 0.0015828, 0.0025740, -0.0006613, 0.0006309
5: 0.0060141, 0.0120894, 0.0058147, 0.0122557, -0.0042974, 0.0040997
6: -0.0015276, 0.0000144, -0.0015698, 0.0000650, -0.0010405, 0.0010907
7: -0.0070900, -0.0031004, -0.0071992, -0.0029694, -0.0026922, 0.0028220
8: -0.0032927, -0.0011946, -0.0033501, -0.0011257, -0.0014158, 0.0014841
9: -0.0004786, 0.0019542, -0.0005585, 0.0020208, -0.0017209, 0.0016417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013214, upper bound: 0.0013055
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013255, upper bound: 0.0013055
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9895564, 0.9931890, 0.9894522, 0.9933334, -0.0024566, 0.0026229
1: -0.0038662, -0.0029611, -0.0038922, -0.0029251, -0.0006121, 0.0006535
2: 0.0056382, 0.0104350, 0.0054475, 0.0105725, -0.0034635, 0.0032439
3: -0.0060227, -0.0038394, -0.0060853, -0.0037526, -0.0014765, 0.0015764
4: 0.0016191, 0.0025476, 0.0015822, 0.0025742, -0.0006703, 0.0006279
5: 0.0060508, 0.0120839, 0.0058110, 0.0122569, -0.0043561, 0.0040800
6: -0.0015262, 0.0000051, -0.0015701, 0.0000659, -0.0010356, 0.0011056
7: -0.0070864, -0.0031245, -0.0071999, -0.0029670, -0.0026793, 0.0028606
8: -0.0032908, -0.0012073, -0.0033505, -0.0011245, -0.0014090, 0.0015044
9: -0.0004640, 0.0019520, -0.0005600, 0.0020212, -0.0017444, 0.0016338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013109, upper bound: 0.0013073
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B1_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013144, upper bound: 0.0013073
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9895155, 0.9933037, 0.9895217, 0.9932467, -0.0025349, 0.0025672
1: -0.0038764, -0.0029325, -0.0038749, -0.0029467, -0.0006316, 0.0006397
2: 0.0054866, 0.0104889, 0.0055619, 0.0104808, -0.0033900, 0.0033474
3: -0.0060472, -0.0037704, -0.0060435, -0.0038047, -0.0015236, 0.0015430
4: 0.0015898, 0.0025580, 0.0016044, 0.0025564, -0.0006561, 0.0006479
5: 0.0058601, 0.0121518, 0.0059549, 0.0121415, -0.0042637, 0.0042101
6: -0.0015434, 0.0000535, -0.0015408, 0.0000294, -0.0010686, 0.0010822
7: -0.0071309, -0.0029993, -0.0071242, -0.0030615, -0.0027647, 0.0027999
8: -0.0033142, -0.0011414, -0.0033107, -0.0011742, -0.0014539, 0.0014725
9: -0.0005403, 0.0019792, -0.0005024, 0.0019751, -0.0017074, 0.0016859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013297, upper bound: 0.0013187
time: 0.83 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013345, upper bound: 0.0013187
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9895160, 0.9932822, 0.9895205, 0.9932485, -0.0025171, 0.0026130
1: -0.0038763, -0.0029379, -0.0038751, -0.0029463, -0.0006272, 0.0006511
2: 0.0055151, 0.0104882, 0.0055596, 0.0104822, -0.0034505, 0.0033238
3: -0.0060469, -0.0037834, -0.0060442, -0.0038036, -0.0015128, 0.0015705
4: 0.0015953, 0.0025579, 0.0016039, 0.0025567, -0.0006678, 0.0006433
5: 0.0058960, 0.0121509, 0.0059520, 0.0121433, -0.0043398, 0.0041804
6: -0.0015432, 0.0000444, -0.0015413, 0.0000301, -0.0010610, 0.0011015
7: -0.0071303, -0.0030228, -0.0071254, -0.0030596, -0.0027452, 0.0028499
8: -0.0033139, -0.0011538, -0.0033113, -0.0011732, -0.0014437, 0.0014987
9: -0.0005259, 0.0019788, -0.0005035, 0.0019758, -0.0017378, 0.0016740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013201, upper bound: 0.0013242
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013241, upper bound: 0.0013242
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9895155, 0.9933037, 0.9894422, 0.9933316, -0.0025765, 0.0026210
1: -0.0038764, -0.0029325, -0.0038947, -0.0029255, -0.0006420, 0.0006531
2: 0.0054866, 0.0104889, 0.0054498, 0.0105857, -0.0034610, 0.0034023
3: -0.0060472, -0.0037704, -0.0060913, -0.0037536, -0.0015486, 0.0015753
4: 0.0015898, 0.0025580, 0.0015827, 0.0025767, -0.0006699, 0.0006585
5: 0.0058601, 0.0121518, 0.0058139, 0.0122735, -0.0043530, 0.0042792
6: -0.0015434, 0.0000535, -0.0015743, 0.0000652, -0.0010861, 0.0011048
7: -0.0071309, -0.0029993, -0.0072108, -0.0029689, -0.0028101, 0.0028585
8: -0.0033142, -0.0011414, -0.0033562, -0.0011255, -0.0014778, 0.0015033
9: -0.0005403, 0.0019792, -0.0005588, 0.0020279, -0.0017431, 0.0017136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013666, upper bound: 0.0013055
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013723, upper bound: 0.0013055
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9895160, 0.9932822, 0.9894415, 0.9933338, -0.0025724, 0.0026592
1: -0.0038763, -0.0029379, -0.0038949, -0.0029250, -0.0006410, 0.0006626
2: 0.0055151, 0.0104882, 0.0054469, 0.0105867, -0.0035115, 0.0033968
3: -0.0060469, -0.0037834, -0.0060917, -0.0037523, -0.0015461, 0.0015983
4: 0.0015953, 0.0025579, 0.0015821, 0.0025769, -0.0006796, 0.0006574
5: 0.0058960, 0.0121509, 0.0058102, 0.0122748, -0.0044165, 0.0042723
6: -0.0015432, 0.0000444, -0.0015746, 0.0000661, -0.0010844, 0.0011210
7: -0.0071303, -0.0030228, -0.0072117, -0.0029665, -0.0028056, 0.0029003
8: -0.0033139, -0.0011538, -0.0033567, -0.0011242, -0.0014754, 0.0015252
9: -0.0005259, 0.0019788, -0.0005603, 0.0020284, -0.0017686, 0.0017108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013551, upper bound: 0.0013073
time: 1.04 seconds

## Relational analysis of IS_A1_A2_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0013073
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9891899, 0.9931320, 0.9895326, 0.9932464, -0.0028819, 0.0024466
1: -0.0039575, -0.0029753, -0.0038721, -0.0029468, -0.0007181, 0.0006096
2: 0.0057135, 0.0109189, 0.0055624, 0.0104663, -0.0032307, 0.0038055
3: -0.0062429, -0.0038737, -0.0060369, -0.0038049, -0.0017321, 0.0014705
4: 0.0016337, 0.0026412, 0.0016045, 0.0025536, -0.0006253, 0.0007365
5: 0.0061456, 0.0126926, 0.0059555, 0.0121233, -0.0040634, 0.0047863
6: -0.0016807, -0.0000190, -0.0015362, 0.0000293, -0.0012148, 0.0010313
7: -0.0074860, -0.0031867, -0.0071122, -0.0030619, -0.0031431, 0.0026684
8: -0.0035010, -0.0012400, -0.0033044, -0.0011744, -0.0016529, 0.0014033
9: -0.0004260, 0.0021957, -0.0005021, 0.0019678, -0.0016272, 0.0019166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A2_A1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012723, upper bound: 0.0013280
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_A2_A1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012735, upper bound: 0.0013280
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9892052, 0.9931027, 0.9895310, 0.9932481, -0.0028464, 0.0024798
1: -0.0039537, -0.0029826, -0.0038726, -0.0029464, -0.0007093, 0.0006179
2: 0.0057520, 0.0108987, 0.0055601, 0.0104685, -0.0032746, 0.0037587
3: -0.0062337, -0.0038912, -0.0060379, -0.0038039, -0.0017108, 0.0014905
4: 0.0016412, 0.0026373, 0.0016040, 0.0025541, -0.0006338, 0.0007275
5: 0.0061939, 0.0126672, 0.0059527, 0.0121261, -0.0041186, 0.0047274
6: -0.0016742, -0.0000313, -0.0015369, 0.0000300, -0.0011999, 0.0010453
7: -0.0074694, -0.0032185, -0.0071141, -0.0030600, -0.0031044, 0.0027046
8: -0.0034922, -0.0012567, -0.0033054, -0.0011734, -0.0016326, 0.0014223
9: -0.0004066, 0.0021855, -0.0005032, 0.0019689, -0.0016493, 0.0018931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A2_A1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0013386
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_A2_A1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0013386
time: 0.88 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9891899, 0.9931320, 0.9894529, 0.9933312, -0.0029309, 0.0024939
1: -0.0039575, -0.0029753, -0.0038920, -0.0029257, -0.0007303, 0.0006214
2: 0.0057135, 0.0109189, 0.0054504, 0.0105716, -0.0032932, 0.0038702
3: -0.0062429, -0.0038737, -0.0060848, -0.0037539, -0.0017615, 0.0014989
4: 0.0016337, 0.0026412, 0.0015828, 0.0025740, -0.0006374, 0.0007491
5: 0.0061456, 0.0126926, 0.0058147, 0.0122557, -0.0041420, 0.0048677
6: -0.0016807, -0.0000190, -0.0015698, 0.0000650, -0.0012355, 0.0010513
7: -0.0074860, -0.0031867, -0.0071992, -0.0029694, -0.0031965, 0.0027200
8: -0.0035010, -0.0012400, -0.0033501, -0.0011257, -0.0016810, 0.0014304
9: -0.0004260, 0.0021957, -0.0005585, 0.0020208, -0.0016586, 0.0019492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013092, upper bound: 0.0013138
time: 0.92 seconds

## Relational analysis of IS_A1_A2_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0013138
time: 0.96 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9892052, 0.9931027, 0.9894522, 0.9933334, -0.0029188, 0.0025199
1: -0.0039537, -0.0029826, -0.0038922, -0.0029251, -0.0007273, 0.0006279
2: 0.0057520, 0.0108987, 0.0054475, 0.0105725, -0.0033275, 0.0038542
3: -0.0062337, -0.0038912, -0.0060853, -0.0037526, -0.0017543, 0.0015145
4: 0.0016412, 0.0026373, 0.0015822, 0.0025742, -0.0006440, 0.0007460
5: 0.0061939, 0.0126672, 0.0058110, 0.0122569, -0.0041851, 0.0048476
6: -0.0016742, -0.0000313, -0.0015701, 0.0000659, -0.0012304, 0.0010622
7: -0.0074694, -0.0032185, -0.0071999, -0.0029670, -0.0031834, 0.0027483
8: -0.0034922, -0.0012567, -0.0033505, -0.0011245, -0.0016741, 0.0014453
9: -0.0004066, 0.0021855, -0.0005600, 0.0020212, -0.0016759, 0.0019412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A2_A1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012994, upper bound: 0.0013167
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B1_A2_A1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013027, upper bound: 0.0013167
time: 0.97 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.9891078, 0.9932055, 0.9895707, 0.9932583, -0.0030240, 0.0024460
1: -0.0039780, -0.0029570, -0.0038626, -0.0029438, -0.0007535, 0.0006095
2: 0.0056163, 0.0110272, 0.0055466, 0.0104160, -0.0032299, 0.0039932
3: -0.0062922, -0.0038294, -0.0060140, -0.0037977, -0.0018175, 0.0014701
4: 0.0016149, 0.0026622, 0.0016014, 0.0025439, -0.0006251, 0.0007729
5: 0.0060233, 0.0128288, 0.0059357, 0.0120600, -0.0040624, 0.0050224
6: -0.0017153, 0.0000121, -0.0015201, 0.0000343, -0.0012747, 0.0010311
7: -0.0075755, -0.0031064, -0.0070706, -0.0030489, -0.0032981, 0.0026677
8: -0.0035480, -0.0011978, -0.0032825, -0.0011675, -0.0017345, 0.0014029
9: -0.0004750, 0.0022503, -0.0005100, 0.0019424, -0.0016267, 0.0020112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0013483
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013061, upper bound: 0.0013483
time: 1.02 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9891098, 0.9932065, 0.9895703, 0.9932399, -0.0030576, 0.0024323
1: -0.0039775, -0.0029567, -0.0038628, -0.0029484, -0.0007619, 0.0006061
2: 0.0056151, 0.0110247, 0.0055709, 0.0104167, -0.0032118, 0.0040375
3: -0.0062911, -0.0038289, -0.0060143, -0.0038088, -0.0018377, 0.0014619
4: 0.0016147, 0.0026617, 0.0016061, 0.0025440, -0.0006216, 0.0007814
5: 0.0060217, 0.0128256, 0.0059662, 0.0120609, -0.0040396, 0.0050781
6: -0.0017144, 0.0000125, -0.0015203, 0.0000265, -0.0012889, 0.0010253
7: -0.0075734, -0.0031054, -0.0070712, -0.0030690, -0.0033347, 0.0026527
8: -0.0035469, -0.0011972, -0.0032828, -0.0011781, -0.0017537, 0.0013950
9: -0.0004756, 0.0022490, -0.0004978, 0.0019427, -0.0016176, 0.0020335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0013386
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0013386
time: 1.00 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9891528, 0.9932300, 0.9894422, 0.9933316, -0.0030264, 0.0025463
1: -0.0039668, -0.0029509, -0.0038947, -0.0029255, -0.0007541, 0.0006345
2: 0.0055841, 0.0109678, 0.0054498, 0.0105857, -0.0033624, 0.0039963
3: -0.0062652, -0.0038147, -0.0060913, -0.0037536, -0.0018190, 0.0015304
4: 0.0016087, 0.0026507, 0.0015827, 0.0025767, -0.0006508, 0.0007735
5: 0.0059827, 0.0127540, 0.0058139, 0.0122735, -0.0042290, 0.0050263
6: -0.0016963, 0.0000223, -0.0015743, 0.0000652, -0.0012757, 0.0010734
7: -0.0075264, -0.0030798, -0.0072108, -0.0029689, -0.0033007, 0.0027771
8: -0.0035222, -0.0011838, -0.0033562, -0.0011255, -0.0017358, 0.0014605
9: -0.0004912, 0.0022203, -0.0005588, 0.0020279, -0.0016935, 0.0020128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A2_A2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0013138
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013562, upper bound: 0.0013138
time: 1.01 seconds

## BFS IS instance: IS_A1_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9891639, 0.9931995, 0.9894415, 0.9933338, -0.0030231, 0.0025733
1: -0.0039640, -0.0029585, -0.0038949, -0.0029250, -0.0007533, 0.0006412
2: 0.0056243, 0.0109532, 0.0054469, 0.0105867, -0.0033980, 0.0039919
3: -0.0062585, -0.0038331, -0.0060917, -0.0037523, -0.0018170, 0.0015466
4: 0.0016165, 0.0026478, 0.0015821, 0.0025769, -0.0006577, 0.0007726
5: 0.0060333, 0.0127356, 0.0058102, 0.0122748, -0.0042738, 0.0050208
6: -0.0016916, 0.0000095, -0.0015746, 0.0000661, -0.0012743, 0.0010847
7: -0.0075143, -0.0031130, -0.0072117, -0.0029665, -0.0032971, 0.0028065
8: -0.0035159, -0.0012013, -0.0033567, -0.0011242, -0.0017339, 0.0014759
9: -0.0004709, 0.0022130, -0.0005603, 0.0020284, -0.0017114, 0.0020105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_A2_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013411, upper bound: 0.0013167
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013459, upper bound: 0.0013167
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9894966, 0.9931965, 0.9893327, 0.9931601, -0.0026498, 0.0027433
1: -0.0038811, -0.0029592, -0.0039220, -0.0029683, -0.0006603, 0.0006836
2: 0.0056282, 0.0105138, 0.0056763, 0.0107304, -0.0036225, 0.0034991
3: -0.0060585, -0.0038348, -0.0061571, -0.0038567, -0.0015926, 0.0016488
4: 0.0016172, 0.0025628, 0.0016265, 0.0026047, -0.0007011, 0.0006772
5: 0.0060383, 0.0121830, 0.0060988, 0.0124554, -0.0045562, 0.0044009
6: -0.0015513, 0.0000082, -0.0016205, -0.0000071, -0.0011170, 0.0011564
7: -0.0071514, -0.0031163, -0.0073303, -0.0031560, -0.0028900, 0.0029920
8: -0.0033250, -0.0012030, -0.0034191, -0.0012239, -0.0015198, 0.0015734
9: -0.0004690, 0.0019917, -0.0004447, 0.0021008, -0.0018245, 0.0017623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0012734
time: 0.98 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0012990
time: 0.97 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9895077, 0.9932835, 0.9893476, 0.9931595, -0.0026602, 0.0028261
1: -0.0038783, -0.0029375, -0.0039182, -0.0029684, -0.0006629, 0.0007042
2: 0.0055133, 0.0104992, 0.0056770, 0.0107106, -0.0037319, 0.0035128
3: -0.0060519, -0.0037825, -0.0061481, -0.0038570, -0.0015989, 0.0016986
4: 0.0015950, 0.0025600, 0.0016267, 0.0026009, -0.0007223, 0.0006799
5: 0.0058937, 0.0121647, 0.0060996, 0.0124306, -0.0046937, 0.0044182
6: -0.0015467, 0.0000449, -0.0016142, -0.0000073, -0.0011214, 0.0011913
7: -0.0071394, -0.0030214, -0.0073140, -0.0031565, -0.0029014, 0.0030823
8: -0.0033187, -0.0011530, -0.0034105, -0.0012241, -0.0015258, 0.0016209
9: -0.0005268, 0.0019843, -0.0004444, 0.0020908, -0.0018796, 0.0017692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0012734
time: 0.98 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0012990
time: 1.00 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9894959, 0.9931975, 0.9893360, 0.9931321, -0.0026852, 0.0027761
1: -0.0038813, -0.0029590, -0.0039211, -0.0029752, -0.0006691, 0.0006917
2: 0.0056269, 0.0105150, 0.0057132, 0.0107260, -0.0036658, 0.0035457
3: -0.0060591, -0.0038342, -0.0061551, -0.0038735, -0.0016139, 0.0016685
4: 0.0016170, 0.0025630, 0.0016337, 0.0026039, -0.0007095, 0.0006863
5: 0.0060366, 0.0121846, 0.0061452, 0.0124499, -0.0046106, 0.0044596
6: -0.0015517, 0.0000087, -0.0016191, -0.0000189, -0.0011319, 0.0011702
7: -0.0071524, -0.0031152, -0.0073267, -0.0031865, -0.0029286, 0.0030277
8: -0.0033256, -0.0012024, -0.0034172, -0.0012399, -0.0015401, 0.0015923
9: -0.0004696, 0.0019923, -0.0004262, 0.0020986, -0.0018463, 0.0017858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0012649
time: 0.93 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0012909
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9895070, 0.9932845, 0.9893517, 0.9931317, -0.0026958, 0.0028490
1: -0.0038785, -0.0029373, -0.0039172, -0.0029753, -0.0006717, 0.0007099
2: 0.0055120, 0.0105002, 0.0057138, 0.0107052, -0.0037621, 0.0035598
3: -0.0060523, -0.0037819, -0.0061457, -0.0038738, -0.0016203, 0.0017123
4: 0.0015947, 0.0025602, 0.0016338, 0.0025999, -0.0007281, 0.0006890
5: 0.0058921, 0.0121659, 0.0061459, 0.0124238, -0.0047317, 0.0044773
6: -0.0015470, 0.0000454, -0.0016125, -0.0000190, -0.0011364, 0.0012010
7: -0.0071402, -0.0030203, -0.0073095, -0.0031869, -0.0029402, 0.0031072
8: -0.0033191, -0.0011525, -0.0034082, -0.0012401, -0.0015462, 0.0016341
9: -0.0005275, 0.0019848, -0.0004259, 0.0020881, -0.0018948, 0.0017929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012648
time: 0.92 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012909
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.9895471, 0.9932157, 0.9892011, 0.9932364, -0.0026593, 0.0028668
1: -0.0038685, -0.0029544, -0.0039548, -0.0029493, -0.0006626, 0.0007143
2: 0.0056029, 0.0104473, 0.0055755, 0.0109041, -0.0037856, 0.0035116
3: -0.0060283, -0.0038233, -0.0062362, -0.0038109, -0.0015983, 0.0017230
4: 0.0016123, 0.0025499, 0.0016070, 0.0026384, -0.0007327, 0.0006797
5: 0.0060064, 0.0120993, 0.0059720, 0.0126740, -0.0047613, 0.0044167
6: -0.0015301, 0.0000163, -0.0016760, 0.0000251, -0.0011210, 0.0012085
7: -0.0070965, -0.0030954, -0.0074738, -0.0030727, -0.0029004, 0.0031267
8: -0.0032961, -0.0011920, -0.0034946, -0.0011801, -0.0015253, 0.0016443
9: -0.0004817, 0.0019582, -0.0004955, 0.0021883, -0.0019066, 0.0017686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0012507
time: 0.95 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0012718
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.9895561, 0.9933034, 0.9892161, 0.9932359, -0.0026698, 0.0029392
1: -0.0038663, -0.0029326, -0.0039510, -0.0029494, -0.0006652, 0.0007324
2: 0.0054870, 0.0104355, 0.0055762, 0.0108843, -0.0038812, 0.0035254
3: -0.0060229, -0.0037706, -0.0062272, -0.0038112, -0.0016046, 0.0017666
4: 0.0015899, 0.0025477, 0.0016072, 0.0026345, -0.0007512, 0.0006823
5: 0.0058607, 0.0120845, 0.0059729, 0.0126491, -0.0048816, 0.0044340
6: -0.0015264, 0.0000533, -0.0016696, 0.0000249, -0.0011254, 0.0012390
7: -0.0070868, -0.0029996, -0.0074575, -0.0030733, -0.0029118, 0.0032057
8: -0.0032910, -0.0011416, -0.0034860, -0.0011804, -0.0015313, 0.0016858
9: -0.0005401, 0.0019522, -0.0004952, 0.0021783, -0.0019548, 0.0017756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013562, upper bound: 0.0012507
time: 1.14 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013562, upper bound: 0.0012718
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.9895473, 0.9931893, 0.9892025, 0.9932380, -0.0026538, 0.0028913
1: -0.0038685, -0.0029610, -0.0039544, -0.0029489, -0.0006613, 0.0007204
2: 0.0056377, 0.0104470, 0.0055735, 0.0109022, -0.0038180, 0.0035043
3: -0.0060281, -0.0038392, -0.0062353, -0.0038099, -0.0015950, 0.0017378
4: 0.0016191, 0.0025499, 0.0016066, 0.0026380, -0.0007390, 0.0006783
5: 0.0060502, 0.0120990, 0.0059695, 0.0126716, -0.0048020, 0.0044075
6: -0.0015300, 0.0000052, -0.0016754, 0.0000257, -0.0011187, 0.0012188
7: -0.0070963, -0.0031241, -0.0074723, -0.0030711, -0.0028943, 0.0031534
8: -0.0032960, -0.0012071, -0.0034937, -0.0011792, -0.0015221, 0.0016583
9: -0.0004642, 0.0019580, -0.0004965, 0.0021873, -0.0019229, 0.0017650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013411, upper bound: 0.0012513
time: 0.92 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013411, upper bound: 0.0012735
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.9895584, 0.9932767, 0.9892176, 0.9932374, -0.0026626, 0.0029677
1: -0.0038657, -0.0029392, -0.0039507, -0.0029490, -0.0006635, 0.0007395
2: 0.0055225, 0.0104323, 0.0055742, 0.0108824, -0.0039188, 0.0035160
3: -0.0060214, -0.0037867, -0.0062263, -0.0038103, -0.0016003, 0.0017837
4: 0.0015968, 0.0025470, 0.0016068, 0.0026342, -0.0007585, 0.0006805
5: 0.0059053, 0.0120805, 0.0059703, 0.0126467, -0.0049288, 0.0044222
6: -0.0015253, 0.0000420, -0.0016690, 0.0000255, -0.0011224, 0.0012510
7: -0.0070841, -0.0030289, -0.0074559, -0.0030716, -0.0029040, 0.0032367
8: -0.0032896, -0.0011570, -0.0034851, -0.0011795, -0.0015272, 0.0017021
9: -0.0005222, 0.0019506, -0.0004962, 0.0021773, -0.0019737, 0.0017708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013459, upper bound: 0.0012513
time: 1.10 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013459, upper bound: 0.0012735
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9894864, 0.9931971, 0.9892896, 0.9932574, -0.0026808, 0.0028306
1: -0.0038837, -0.0029590, -0.0039327, -0.0029440, -0.0006680, 0.0007053
2: 0.0056274, 0.0105274, 0.0055478, 0.0107873, -0.0037378, 0.0035400
3: -0.0060647, -0.0038344, -0.0061830, -0.0037982, -0.0016113, 0.0017013
4: 0.0016171, 0.0025654, 0.0016017, 0.0026157, -0.0007234, 0.0006852
5: 0.0060372, 0.0122002, 0.0059371, 0.0125270, -0.0047012, 0.0044524
6: -0.0015557, 0.0000085, -0.0016387, 0.0000339, -0.0011301, 0.0011932
7: -0.0071627, -0.0031155, -0.0073773, -0.0030499, -0.0029238, 0.0030872
8: -0.0033309, -0.0012026, -0.0034438, -0.0011680, -0.0015376, 0.0016235
9: -0.0004694, 0.0019985, -0.0005095, 0.0021294, -0.0018826, 0.0017829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0013201
time: 1.23 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0013482
time: 0.91 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.70 seconds
IS_A1_A1_A1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013577, upper bound: 0.0013046
IS_A1_A1_A1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013613, upper bound: 0.0013026
IS_A1_A1_A1_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013577, upper bound: 0.0013046
IS_A1_A1_A1_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013613, upper bound: 0.0013026
IS_A1_A1_A1_B1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013518, upper bound: 0.0012962
IS_A1_A1_A1_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013569, upper bound: 0.0012943
IS_A1_A1_A1_B1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013518, upper bound: 0.0012962
IS_A1_A1_A1_B1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013569, upper bound: 0.0012943
IS_A1_A1_A1_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0013635
IS_A1_A1_A1_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013532, upper bound: 0.0013532
IS_A1_A1_A1_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0013635
IS_A1_A1_A1_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013532, upper bound: 0.0013531
IS_A1_A1_A1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014033, upper bound: 0.0013381
IS_A1_A1_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013918, upper bound: 0.0013384
IS_A1_A1_A1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014033, upper bound: 0.0013381
IS_A1_A1_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013918, upper bound: 0.0013384
IS_A1_A1_A1_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013593, upper bound: 0.0013107
IS_A1_A1_A1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013028
IS_A1_A1_A1_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013593, upper bound: 0.0013107
IS_A1_A1_A1_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013028
IS_A1_A1_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0012870
IS_A1_A1_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0012870
IS_A1_A1_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0012870
IS_A1_A1_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0012870
IS_A1_A1_A1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013792, upper bound: 0.0013417
IS_A1_A1_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013445
IS_A1_A1_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013792, upper bound: 0.0013417
IS_A1_A1_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013681, upper bound: 0.0013445
IS_A1_A1_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0013304
IS_A1_A1_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0013306
IS_A1_A1_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014198, upper bound: 0.0013304
IS_A1_A1_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0014087, upper bound: 0.0013306
IS_A1_A1_A2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013419, upper bound: 0.0013340
IS_A1_A1_A2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013419, upper bound: 0.0013340
IS_A1_A1_A2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0013244
IS_A1_A1_A2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0013244
IS_A1_A1_A2_B1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013709, upper bound: 0.0013079
IS_A1_A1_A2_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013709, upper bound: 0.0013079
IS_A1_A1_A2_B1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013843, upper bound: 0.0013051
IS_A1_A1_A2_B1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013843, upper bound: 0.0013051
IS_A1_A1_A2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013436, upper bound: 0.0013313
IS_A1_A1_A2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013436, upper bound: 0.0013313
IS_A1_A1_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013478, upper bound: 0.0013222
IS_A1_A1_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013478, upper bound: 0.0013222
IS_A1_A1_A2_B1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013543, upper bound: 0.0012656
IS_A1_A1_A2_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013544, upper bound: 0.0012635
IS_A1_A1_A2_B1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013771, upper bound: 0.0012635
IS_A1_A1_A2_B1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0012618
IS_A1_A1_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013419, upper bound: 0.0013798
IS_A1_A1_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013419, upper bound: 0.0013798
IS_A1_A1_A2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0013682
IS_A1_A1_A2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013456, upper bound: 0.0013682
IS_A1_A1_A2_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0013309
IS_A1_A1_A2_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0013362
IS_A1_A1_A2_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0013227
IS_A1_A1_A2_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0013269
IS_A1_A1_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013757, upper bound: 0.0013015
IS_A1_A1_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013757, upper bound: 0.0013058
IS_A1_A1_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013675, upper bound: 0.0013037
IS_A1_A1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013675, upper bound: 0.0013073
IS_A1_A1_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013007
IS_A1_A1_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013040
IS_A1_A1_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0013020
IS_A1_A1_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013771, upper bound: 0.0013052
IS_A1_A2_B1_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012848, upper bound: 0.0013187
IS_A1_A2_B1_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012859, upper bound: 0.0013187
IS_A1_A2_B1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012767, upper bound: 0.0013242
IS_A1_A2_B1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012772, upper bound: 0.0013242
IS_A1_A2_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013214, upper bound: 0.0013055
IS_A1_A2_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013255, upper bound: 0.0013055
IS_A1_A2_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013109, upper bound: 0.0013073
IS_A1_A2_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013144, upper bound: 0.0013073
IS_A1_A2_B1_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013297, upper bound: 0.0013187
IS_A1_A2_B1_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013345, upper bound: 0.0013187
IS_A1_A2_B1_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013201, upper bound: 0.0013242
IS_A1_A2_B1_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013241, upper bound: 0.0013242
IS_A1_A2_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013666, upper bound: 0.0013055
IS_A1_A2_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013723, upper bound: 0.0013055
IS_A1_A2_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013551, upper bound: 0.0013073
IS_A1_A2_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0013073
IS_A1_A2_B1_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012723, upper bound: 0.0013280
IS_A1_A2_B1_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012735, upper bound: 0.0013280
IS_A1_A2_B1_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0013386
IS_A1_A2_B1_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0013386
IS_A1_A2_B1_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013092, upper bound: 0.0013138
IS_A1_A2_B1_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0013138
IS_A1_A2_B1_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0012994, upper bound: 0.0013167
IS_A1_A2_B1_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013027, upper bound: 0.0013167
IS_A1_A2_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0013483
IS_A1_A2_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013061, upper bound: 0.0013483
IS_A1_A2_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0013386
IS_A1_A2_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0013386
IS_A1_A2_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0013138
IS_A1_A2_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013562, upper bound: 0.0013138
IS_A1_A2_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013411, upper bound: 0.0013167
IS_A1_A2_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013459, upper bound: 0.0013167
IS_A1_A2_B2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0012734
IS_A1_A2_B2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0012990
IS_A1_A2_B2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0012734
IS_A1_A2_B2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0012990
IS_A1_A2_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0012649
IS_A1_A2_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0012909
IS_A1_A2_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012648
IS_A1_A2_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013112, upper bound: 0.0012909
IS_A1_A2_B2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0012507
IS_A1_A2_B2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0012718
IS_A1_A2_B2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013562, upper bound: 0.0012507
IS_A1_A2_B2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013562, upper bound: 0.0012718
IS_A1_A2_B2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013411, upper bound: 0.0012513
IS_A1_A2_B2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013411, upper bound: 0.0012735
IS_A1_A2_B2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013459, upper bound: 0.0012513
IS_A1_A2_B2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013459, upper bound: 0.0012735
IS_A1_A2_B2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0013201
IS_A1_A2_B2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.70
Output dim: 0, lower bound: -0.0013012, upper bound: 0.0013482
IS_A1_A2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013280, upper bound: 0.0013482
IS_A1_A2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013355, upper bound: 0.0013386
IS_A1_A2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013386, upper bound: 0.0013386
IS_A1_A2_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013096
IS_A1_A2_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013839, upper bound: 0.0013138
IS_A1_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013771, upper bound: 0.0013133
IS_A1_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013770, upper bound: 0.0013167
IS_A2_A1_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
IS_A2_A1_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
IS_A2_A1_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
IS_A2_A1_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0013879
IS_A2_A1_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
IS_A2_A1_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
IS_A2_A1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
IS_A2_A1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013734, upper bound: 0.0014312
IS_A2_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013808
IS_A2_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013809
IS_A2_A1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013808
IS_A2_A1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0013809
IS_A2_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
IS_A2_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
IS_A2_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014223
IS_A2_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013830, upper bound: 0.0014224
IS_A2_A1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013045, upper bound: 0.0014206
IS_A2_A1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013050, upper bound: 0.0014091
IS_A2_A1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013079, upper bound: 0.0013918
IS_A2_A1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013050, upper bound: 0.0014091
IS_A2_A1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013140, upper bound: 0.0014166
IS_A2_A1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013143, upper bound: 0.0014068
IS_A2_A1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013189, upper bound: 0.0013901
IS_A2_A1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013143, upper bound: 0.0014068
IS_A2_A1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013466, upper bound: 0.0014206
IS_A2_A1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013475, upper bound: 0.0014091
IS_A2_A1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013525, upper bound: 0.0013918
IS_A2_A1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013475, upper bound: 0.0014091
IS_A2_A1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014166
IS_A2_A1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014068
IS_A2_A1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013605, upper bound: 0.0013901
IS_A2_A1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013555, upper bound: 0.0014068
IS_A2_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013010, upper bound: 0.0013788
IS_A2_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0012990, upper bound: 0.0013940
IS_A2_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013010, upper bound: 0.0013787
IS_A2_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0012990, upper bound: 0.0013940
IS_A2_A2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014067
IS_A2_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013403, upper bound: 0.0013939
IS_A2_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013366, upper bound: 0.0014011
IS_A2_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013430, upper bound: 0.0014011
IS_A2_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014299
IS_A2_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014185
IS_A2_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0012925, upper bound: 0.0013971
IS_A2_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0012908, upper bound: 0.0014185
IS_A2_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013306, upper bound: 0.0014299
IS_A2_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013318, upper bound: 0.0014185
IS_A2_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013253, upper bound: 0.0014178
IS_A2_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013304, upper bound: 0.0014178
IS_A2_A2_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013381
IS_A2_A2_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013381
IS_A2_A2_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013294
IS_A2_A2_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013295
IS_A2_A2_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013381
IS_A2_A2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013381
IS_A2_A2_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013295
IS_A2_A2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013295
IS_A2_A2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013839
IS_A2_A2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013839
IS_A2_A2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013771
IS_A2_A2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013770
IS_A2_A2_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013839
IS_A2_A2_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013138, upper bound: 0.0013839
IS_A2_A2_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013770
IS_A2_A2_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 0, lower bound: -0.0013167, upper bound: 0.0013770

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.99 + 598.41 = 601.40 seconds
