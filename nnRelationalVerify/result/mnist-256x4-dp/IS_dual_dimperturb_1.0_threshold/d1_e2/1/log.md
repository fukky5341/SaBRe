## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00158454


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0050904, 0.0010837, -0.0050904, 0.0010837, -0.0061741, 0.0061741)
1: (0.0029200, 0.0073178, 0.0029200, 0.0073178, -0.0043977, 0.0043977)
2: (0.0085237, 0.0201408, 0.0085237, 0.0201408, -0.0107039, 0.0107039)
3: (-0.0071492, -0.0019233, -0.0071492, -0.0019233, -0.0052259, 0.0052259)
4: (0.0037488, 0.0054888, 0.0037488, 0.0054888, -0.0016757, 0.0016757)
5: (-0.0040829, -0.0002159, -0.0040829, -0.0002159, -0.0038670, 0.0038670)
6: (-0.0068849, -0.0050096, -0.0068849, -0.0050096, -0.0018753, 0.0018753)
7: (-0.0038616, -0.0001440, -0.0038616, -0.0001440, -0.0037175, 0.0037175)
8: (-0.0077969, -0.0001072, -0.0077969, -0.0001072, -0.0076897, 0.0076897)
9: (1.0003322, 1.0021871, 1.0003322, 1.0021871, -0.0018549, 0.0018549)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 2.73 = 3.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0017606, upper bound: 0.0017606

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017606
time: 1.94 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
time: 2.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.19 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 4.19
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017606
IS_B2, status: Status.UNKNOWN, split count: 1, time: 4.19
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0050904, 0.0010837, -0.0050730, 0.0008325, -0.0059229, 0.0061567
1: 0.0029200, 0.0073178, 0.0031296, 0.0073050, -0.0043850, 0.0041881
2: 0.0085237, 0.0201408, 0.0085573, 0.0196800, -0.0102578, 0.0106694
3: -0.0071492, -0.0019233, -0.0069078, -0.0019382, -0.0052110, 0.0049845
4: 0.0037488, 0.0054888, 0.0038273, 0.0054838, -0.0016540, 0.0015904
5: -0.0040829, -0.0002159, -0.0039506, -0.0002278, -0.0038551, 0.0037348
6: -0.0068849, -0.0050096, -0.0068052, -0.0050149, -0.0018701, 0.0017956
7: -0.0038616, -0.0001440, -0.0038496, -0.0003388, -0.0035228, 0.0037056
8: -0.0077969, -0.0001072, -0.0074959, -0.0001297, -0.0076673, 0.0073886
9: 1.0003322, 1.0021871, 1.0003341, 1.0021654, -0.0018332, 0.0018530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
time: 1.81 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
time: 1.52 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0050885, 0.0010356, -0.0052048, 0.0007401, -0.0058287, 0.0062404
1: 0.0029572, 0.0073165, 0.0031823, 0.0074221, -0.0044649, 0.0041342
2: 0.0085274, 0.0200529, 0.0083194, 0.0195125, -0.0101865, 0.0108410
3: -0.0071050, -0.0019249, -0.0068315, -0.0018053, -0.0052997, 0.0049066
4: 0.0037636, 0.0054883, 0.0038542, 0.0055269, -0.0016979, 0.0016340
5: -0.0040548, -0.0002172, -0.0038910, -0.0001663, -0.0038885, 0.0036738
6: -0.0068698, -0.0050102, -0.0067760, -0.0049719, -0.0018979, 0.0017658
7: -0.0038603, -0.0001770, -0.0039687, -0.0003721, -0.0034882, 0.0037918
8: -0.0077394, -0.0001097, -0.0073862, 0.0000242, -0.0077636, 0.0072765
9: 1.0003325, 1.0021849, 1.0003170, 1.0023339, -0.0020014, 0.0018679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016981
time: 1.51 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017419, upper bound: 0.0017419
time: 1.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.60 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016981
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 9, lower bound: -0.0017419, upper bound: 0.0017419

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.0050730, 0.0008325, -0.0050730, 0.0008325, -0.0059055, 0.0059055
1: 0.0031296, 0.0073050, 0.0031296, 0.0073050, -0.0041754, 0.0041754
2: 0.0085573, 0.0196800, 0.0085573, 0.0196800, -0.0102233, 0.0102233
3: -0.0069078, -0.0019382, -0.0069078, -0.0019382, -0.0049696, 0.0049696
4: 0.0038273, 0.0054838, 0.0038273, 0.0054838, -0.0015687, 0.0015687
5: -0.0039506, -0.0002278, -0.0039506, -0.0002278, -0.0037228, 0.0037228
6: -0.0068052, -0.0050149, -0.0068052, -0.0050149, -0.0017904, 0.0017904
7: -0.0038496, -0.0003388, -0.0038496, -0.0003388, -0.0035108, 0.0035108
8: -0.0074959, -0.0001297, -0.0074959, -0.0001297, -0.0073662, 0.0073662
9: 1.0003341, 1.0021654, 1.0003341, 1.0021654, -0.0018313, 0.0018313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016981, upper bound: 0.0015630
time: 1.17 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017420, upper bound: 0.0017429
time: 1.74 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -0.0052048, 0.0007401, -0.0050730, 0.0008325, -0.0060373, 0.0058131
1: 0.0031823, 0.0074221, 0.0031296, 0.0073050, -0.0041227, 0.0042924
2: 0.0083194, 0.0195125, 0.0085573, 0.0196800, -0.0104818, 0.0100740
3: -0.0068315, -0.0018053, -0.0069078, -0.0019382, -0.0048933, 0.0051024
4: 0.0038542, 0.0055269, 0.0038273, 0.0054838, -0.0015659, 0.0016261
5: -0.0038910, -0.0001663, -0.0039506, -0.0002278, -0.0036632, 0.0037843
6: -0.0067760, -0.0049719, -0.0068052, -0.0050149, -0.0017612, 0.0018333
7: -0.0039687, -0.0003721, -0.0038496, -0.0003388, -0.0036299, 0.0034775
8: -0.0073862, 0.0000242, -0.0074959, -0.0001297, -0.0072566, 0.0075201
9: 1.0003170, 1.0023339, 1.0003341, 1.0021654, -0.0018485, 0.0019997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016981, upper bound: 0.0015630
time: 1.29 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017420, upper bound: 0.0017430
time: 2.11 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -0.0050293, 0.0010325, -0.0047391, 0.0005960, -0.0056253, 0.0057715
1: 0.0029621, 0.0072762, 0.0032949, 0.0071035, -0.0041414, 0.0039813
2: 0.0086381, 0.0200468, 0.0091941, 0.0192440, -0.0097854, 0.0099597
3: -0.0071023, -0.0019742, -0.0067071, -0.0021938, -0.0049085, 0.0047329
4: 0.0037645, 0.0054718, 0.0038956, 0.0053966, -0.0015771, 0.0015762
5: -0.0040520, -0.0002535, -0.0037981, -0.0004521, -0.0035999, 0.0035446
6: -0.0068689, -0.0050281, -0.0067318, -0.0051139, -0.0017550, 0.0017038
7: -0.0038314, -0.0001828, -0.0037385, -0.0004570, -0.0033744, 0.0035557
8: -0.0077353, -0.0001829, -0.0072069, -0.0005544, -0.0071809, 0.0070240
9: 1.0003378, 1.0021112, 1.0003587, 1.0017505, -0.0014126, 0.0017525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016981
time: 1.48 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016981
time: 1.67 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -0.0050885, 0.0010356, -0.0051029, 0.0007293, -0.0058179, 0.0061385
1: 0.0029572, 0.0073165, 0.0031957, 0.0073574, -0.0044002, 0.0041208
2: 0.0085274, 0.0200529, 0.0085136, 0.0194918, -0.0101662, 0.0104490
3: -0.0071050, -0.0019249, -0.0068226, -0.0018850, -0.0052200, 0.0048977
4: 0.0037636, 0.0054883, 0.0038572, 0.0054998, -0.0016157, 0.0016311
5: -0.0040548, -0.0002172, -0.0038828, -0.0002325, -0.0038223, 0.0036655
6: -0.0068698, -0.0050102, -0.0067728, -0.0050023, -0.0018675, 0.0017626
7: -0.0038603, -0.0001770, -0.0039171, -0.0003870, -0.0034733, 0.0037401
8: -0.0077394, -0.0001097, -0.0073722, -0.0001054, -0.0076340, 0.0072625
9: 1.0003325, 1.0021849, 1.0003257, 1.0022068, -0.0018743, 0.0018592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017420, upper bound: 0.0017419
time: 1.80 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017420, upper bound: 0.0017419
time: 1.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.67 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0016981, upper bound: 0.0015630
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0017420, upper bound: 0.0017429
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0016981, upper bound: 0.0015630
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0017420, upper bound: 0.0017430
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016981
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016981
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0017420, upper bound: 0.0017419
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.67
Output dim: 9, lower bound: -0.0017420, upper bound: 0.0017419

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0045952, 0.0006857, -0.0050137, 0.0008294, -0.0054246, 0.0056994
1: 0.0032486, 0.0069791, 0.0031346, 0.0072647, -0.0040161, 0.0038444
2: 0.0094526, 0.0194093, 0.0086682, 0.0196740, -0.0093246, 0.0098133
3: -0.0067785, -0.0023363, -0.0069052, -0.0019877, -0.0047908, 0.0045689
4: 0.0038708, 0.0053503, 0.0038281, 0.0054673, -0.0015202, 0.0014452
5: -0.0038524, -0.0005188, -0.0039478, -0.0002641, -0.0035882, 0.0034290
6: -0.0067591, -0.0051604, -0.0068043, -0.0050328, -0.0017263, 0.0016439
7: -0.0036138, -0.0004292, -0.0038205, -0.0003449, -0.0032689, 0.0033913
8: -0.0073157, -0.0007215, -0.0074916, -0.0002029, -0.0071128, 0.0067702
9: 1.0003768, 1.0015694, 1.0003395, 1.0020918, -0.0017149, 0.0012299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015587, upper bound: 0.0015588
time: 1.02 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015587, upper bound: 0.0015632
time: 1.14 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0049766, 0.0008226, -0.0050730, 0.0008325, -0.0058091, 0.0058956
1: 0.0031411, 0.0072423, 0.0031296, 0.0073050, -0.0041639, 0.0041126
2: 0.0087394, 0.0196612, 0.0085573, 0.0196800, -0.0098260, 0.0102048
3: -0.0068998, -0.0020146, -0.0069078, -0.0019382, -0.0049616, 0.0048932
4: 0.0038299, 0.0054582, 0.0038273, 0.0054838, -0.0015671, 0.0014793
5: -0.0039430, -0.0002896, -0.0039506, -0.0002278, -0.0037152, 0.0036610
6: -0.0068023, -0.0050436, -0.0068052, -0.0050149, -0.0017875, 0.0017616
7: -0.0038014, -0.0003530, -0.0038496, -0.0003388, -0.0034626, 0.0034966
8: -0.0074830, -0.0002510, -0.0074959, -0.0001297, -0.0073534, 0.0072448
9: 1.0003421, 1.0020459, 1.0003341, 1.0021654, -0.0018233, 0.0017117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015632, upper bound: 0.0016989
time: 1.56 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015632, upper bound: 0.0017430
time: 1.23 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0047391, 0.0005960, -0.0050137, 0.0008294, -0.0055684, 0.0056096
1: 0.0032949, 0.0071035, 0.0031346, 0.0072647, -0.0039698, 0.0039689
2: 0.0091941, 0.0192440, 0.0086682, 0.0196740, -0.0096006, 0.0096829
3: -0.0067071, -0.0021938, -0.0069052, -0.0019877, -0.0047194, 0.0047114
4: 0.0038956, 0.0053966, 0.0038281, 0.0054673, -0.0015280, 0.0015053
5: -0.0037981, -0.0004521, -0.0039478, -0.0002641, -0.0035339, 0.0034958
6: -0.0067318, -0.0051139, -0.0068043, -0.0050328, -0.0016991, 0.0016904
7: -0.0037385, -0.0004570, -0.0038205, -0.0003449, -0.0033936, 0.0033635
8: -0.0072069, -0.0005544, -0.0074916, -0.0002029, -0.0070040, 0.0069372
9: 1.0003587, 1.0017505, 1.0003395, 1.0020918, -0.0017331, 0.0014110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015585
time: 1.19 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015630
time: 1.20 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0051029, 0.0007293, -0.0050730, 0.0008325, -0.0059354, 0.0058023
1: 0.0031957, 0.0073574, 0.0031296, 0.0073050, -0.0041093, 0.0042278
2: 0.0085136, 0.0194918, 0.0085573, 0.0196800, -0.0100836, 0.0100529
3: -0.0068226, -0.0018850, -0.0069078, -0.0019382, -0.0048844, 0.0050228
4: 0.0038572, 0.0054998, 0.0038273, 0.0054838, -0.0015636, 0.0015384
5: -0.0038828, -0.0002325, -0.0039506, -0.0002278, -0.0036550, 0.0037181
6: -0.0067728, -0.0050023, -0.0068052, -0.0050149, -0.0017580, 0.0018029
7: -0.0039171, -0.0003870, -0.0038496, -0.0003388, -0.0035783, 0.0034626
8: -0.0073722, -0.0001054, -0.0074959, -0.0001297, -0.0072425, 0.0073904
9: 1.0003257, 1.0022068, 1.0003341, 1.0021654, -0.0018398, 0.0018727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016988
time: 1.28 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0017430
time: 1.32 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0050137, 0.0008294, -0.0047391, 0.0005960, -0.0056096, 0.0055684
1: 0.0031346, 0.0072647, 0.0032949, 0.0071035, -0.0039689, 0.0039698
2: 0.0086682, 0.0196740, 0.0091941, 0.0192440, -0.0096829, 0.0096007
3: -0.0069052, -0.0019877, -0.0067071, -0.0021938, -0.0047114, 0.0047194
4: 0.0038281, 0.0054673, 0.0038956, 0.0053966, -0.0015053, 0.0015280
5: -0.0039478, -0.0002641, -0.0037981, -0.0004521, -0.0034958, 0.0035339
6: -0.0068043, -0.0050328, -0.0067318, -0.0051139, -0.0016904, 0.0016991
7: -0.0038205, -0.0003449, -0.0037385, -0.0004570, -0.0033635, 0.0033936
8: -0.0074916, -0.0002029, -0.0072069, -0.0005544, -0.0069372, 0.0070040
9: 1.0003395, 1.0020918, 1.0003587, 1.0017505, -0.0014110, 0.0017331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.09 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.18 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0051472, 0.0007367, -0.0047391, 0.0005960, -0.0057432, 0.0054758
1: 0.0031881, 0.0073828, 0.0032949, 0.0071035, -0.0039154, 0.0040879
2: 0.0084274, 0.0195058, 0.0091941, 0.0192440, -0.0098467, 0.0093632
3: -0.0068286, -0.0018535, -0.0067071, -0.0021938, -0.0046348, 0.0048536
4: 0.0038551, 0.0055108, 0.0038956, 0.0053966, -0.0014951, 0.0015696
5: -0.0038880, -0.0002017, -0.0037981, -0.0004521, -0.0034360, 0.0035964
6: -0.0067750, -0.0049895, -0.0067318, -0.0051139, -0.0016611, 0.0017424
7: -0.0039401, -0.0003782, -0.0037385, -0.0004570, -0.0034831, 0.0033603
8: -0.0073816, -0.0000473, -0.0072069, -0.0005544, -0.0068272, 0.0071596
9: 1.0003222, 1.0022615, 1.0003587, 1.0017505, -0.0014282, 0.0019028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.11 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.15 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0050730, 0.0008325, -0.0051029, 0.0007293, -0.0058023, 0.0059354
1: 0.0031296, 0.0073050, 0.0031957, 0.0073574, -0.0042278, 0.0041093
2: 0.0085573, 0.0196800, 0.0085136, 0.0194918, -0.0100529, 0.0100836
3: -0.0069078, -0.0019382, -0.0068226, -0.0018850, -0.0050228, 0.0048844
4: 0.0038273, 0.0054838, 0.0038572, 0.0054998, -0.0015384, 0.0015636
5: -0.0039506, -0.0002278, -0.0038828, -0.0002325, -0.0037181, 0.0036550
6: -0.0068052, -0.0050149, -0.0067728, -0.0050023, -0.0018029, 0.0017580
7: -0.0038496, -0.0003388, -0.0039171, -0.0003870, -0.0034626, 0.0035783
8: -0.0074959, -0.0001297, -0.0073722, -0.0001054, -0.0073904, 0.0072425
9: 1.0003341, 1.0021654, 1.0003257, 1.0022068, -0.0018727, 0.0018398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015626
time: 1.37 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.28 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0052048, 0.0007401, -0.0051029, 0.0007293, -0.0059341, 0.0058430
1: 0.0031823, 0.0074221, 0.0031957, 0.0073574, -0.0041751, 0.0042264
2: 0.0083194, 0.0195125, 0.0085136, 0.0194918, -0.0102257, 0.0098362
3: -0.0068315, -0.0018053, -0.0068226, -0.0018850, -0.0049465, 0.0050172
4: 0.0038542, 0.0055269, 0.0038572, 0.0054998, -0.0015238, 0.0016170
5: -0.0038910, -0.0001663, -0.0038828, -0.0002325, -0.0036585, 0.0037164
6: -0.0067760, -0.0049719, -0.0067728, -0.0050023, -0.0017737, 0.0018009
7: -0.0039687, -0.0003721, -0.0039171, -0.0003870, -0.0035817, 0.0035449
8: -0.0073862, 0.0000242, -0.0073722, -0.0001054, -0.0072808, 0.0073964
9: 1.0003170, 1.0023339, 1.0003257, 1.0022068, -0.0018898, 0.0020082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015626
time: 1.51 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.95 seconds
IS_B1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015587, upper bound: 0.0015588
IS_B1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015587, upper bound: 0.0015632
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015632, upper bound: 0.0016989
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015632, upper bound: 0.0017430
IS_B1_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015585
IS_B1_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015630
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016988
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0017430
IS_B2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
IS_B2_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
IS_B2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
IS_B2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
IS_B2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015626
IS_B2_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
IS_B2_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015626
IS_B2_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049766, 0.0008226, -0.0045952, 0.0006857, -0.0056623, 0.0054179
1: 0.0031411, 0.0072423, 0.0032486, 0.0069791, -0.0038380, 0.0039937
2: 0.0087394, 0.0196612, 0.0094526, 0.0194093, -0.0097445, 0.0093123
3: -0.0068998, -0.0020146, -0.0067785, -0.0023363, -0.0045636, 0.0047639
4: 0.0038299, 0.0054582, 0.0038708, 0.0053503, -0.0014441, 0.0015081
5: -0.0039430, -0.0002896, -0.0038524, -0.0005188, -0.0034242, 0.0035628
6: -0.0068023, -0.0050436, -0.0067591, -0.0051604, -0.0016419, 0.0017155
7: -0.0038014, -0.0003530, -0.0036138, -0.0004292, -0.0033722, 0.0032608
8: -0.0074830, -0.0002510, -0.0073157, -0.0007215, -0.0067615, 0.0070647
9: 1.0003421, 1.0020459, 1.0003768, 1.0015694, -0.0012273, 0.0016690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015202, upper bound: 0.0016209
time: 1.71 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015358, upper bound: 0.0016653
time: 1.54 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049766, 0.0008226, -0.0049766, 0.0008226, -0.0057992, 0.0057992
1: 0.0031411, 0.0072423, 0.0031411, 0.0072423, -0.0041012, 0.0041012
2: 0.0087394, 0.0196612, 0.0087394, 0.0196612, -0.0098070, 0.0098070
3: -0.0068998, -0.0020146, -0.0068998, -0.0020146, -0.0048853, 0.0048853
4: 0.0038299, 0.0054582, 0.0038299, 0.0054582, -0.0014776, 0.0014776
5: -0.0039430, -0.0002896, -0.0039430, -0.0002896, -0.0036534, 0.0036534
6: -0.0068023, -0.0050436, -0.0068023, -0.0050436, -0.0017587, 0.0017587
7: -0.0038014, -0.0003530, -0.0038014, -0.0003530, -0.0034484, 0.0034484
8: -0.0074830, -0.0002510, -0.0074830, -0.0002510, -0.0072320, 0.0072320
9: 1.0003421, 1.0020459, 1.0003421, 1.0020459, -0.0017037, 0.0017037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015202, upper bound: 0.0016209
time: 1.16 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015358, upper bound: 0.0017190
time: 1.65 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0051029, 0.0007293, -0.0045952, 0.0006857, -0.0057886, 0.0053246
1: 0.0031957, 0.0073574, 0.0032486, 0.0069791, -0.0037834, 0.0041088
2: 0.0085136, 0.0194918, 0.0094526, 0.0194093, -0.0099986, 0.0091604
3: -0.0068226, -0.0018850, -0.0067785, -0.0023363, -0.0044863, 0.0048935
4: 0.0038572, 0.0054998, 0.0038708, 0.0053503, -0.0014407, 0.0015621
5: -0.0038828, -0.0002325, -0.0038524, -0.0005188, -0.0033640, 0.0036199
6: -0.0067728, -0.0050023, -0.0067591, -0.0051604, -0.0016124, 0.0017567
7: -0.0039171, -0.0003870, -0.0036138, -0.0004292, -0.0034879, 0.0032267
8: -0.0073722, -0.0001054, -0.0073157, -0.0007215, -0.0066507, 0.0072102
9: 1.0003257, 1.0022068, 1.0003768, 1.0015694, -0.0012437, 0.0018300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016202
time: 1.78 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016653
time: 2.12 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0051029, 0.0007293, -0.0049766, 0.0008226, -0.0059255, 0.0057059
1: 0.0031957, 0.0073574, 0.0031411, 0.0072423, -0.0040466, 0.0042163
2: 0.0085136, 0.0194918, 0.0087394, 0.0196612, -0.0100645, 0.0096627
3: -0.0068226, -0.0018850, -0.0068998, -0.0020146, -0.0048080, 0.0050149
4: 0.0038572, 0.0054998, 0.0038299, 0.0054582, -0.0014852, 0.0015367
5: -0.0038828, -0.0002325, -0.0039430, -0.0002896, -0.0035932, 0.0037105
6: -0.0067728, -0.0050023, -0.0068023, -0.0050436, -0.0017293, 0.0018000
7: -0.0039171, -0.0003870, -0.0038014, -0.0003530, -0.0035641, 0.0034144
8: -0.0073722, -0.0001054, -0.0074830, -0.0002510, -0.0071212, 0.0073776
9: 1.0003257, 1.0022068, 1.0003421, 1.0020459, -0.0017202, 0.0018647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016726
time: 1.48 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015353, upper bound: 0.0016653
time: 1.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.03 seconds
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 9, lower bound: -0.0015202, upper bound: 0.0016209
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 9, lower bound: -0.0015358, upper bound: 0.0016653
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 9, lower bound: -0.0015202, upper bound: 0.0016209
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 9, lower bound: -0.0015358, upper bound: 0.0017190
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016202
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016653
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016726
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 9, lower bound: -0.0015353, upper bound: 0.0016653

## BFS IS instance: IS_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0045660, 0.0006846, -0.0053864, 0.0053390
1: 0.0032021, 0.0070349, 0.0032505, 0.0069565, -0.0037544, 0.0037844
2: 0.0092501, 0.0195841, 0.0095071, 0.0194070, -0.0092320, 0.0091406
3: -0.0068406, -0.0022609, -0.0067778, -0.0023629, -0.0044776, 0.0045169
4: 0.0038494, 0.0053766, 0.0038710, 0.0053415, -0.0014221, 0.0014314
5: -0.0039186, -0.0004460, -0.0038510, -0.0005358, -0.0033828, 0.0034049
6: -0.0067835, -0.0051295, -0.0067588, -0.0051695, -0.0016140, 0.0016293
7: -0.0036297, -0.0004172, -0.0035949, -0.0004315, -0.0031982, 0.0031777
8: -0.0074341, -0.0005859, -0.0073140, -0.0007574, -0.0066767, 0.0067281
9: 1.0003712, 1.0016996, 1.0003799, 1.0015326, -0.0011613, 0.0013196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014941, upper bound: 0.0015525
time: 1.43 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014464, upper bound: 0.0015370
time: 1.25 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0045952, 0.0006857, -0.0055329, 0.0054111
1: 0.0031500, 0.0071585, 0.0032486, 0.0069791, -0.0038291, 0.0039099
2: 0.0089850, 0.0196473, 0.0094526, 0.0194093, -0.0094227, 0.0092968
3: -0.0068959, -0.0021200, -0.0067785, -0.0023363, -0.0045597, 0.0046584
4: 0.0038312, 0.0054226, 0.0038708, 0.0053503, -0.0014423, 0.0014449
5: -0.0039354, -0.0003724, -0.0038524, -0.0005188, -0.0034166, 0.0034800
6: -0.0068006, -0.0050825, -0.0067591, -0.0051604, -0.0016402, 0.0016765
7: -0.0037344, -0.0003632, -0.0036138, -0.0004292, -0.0033052, 0.0032506
8: -0.0074731, -0.0004146, -0.0073157, -0.0007215, -0.0067516, 0.0069011
9: 1.0003535, 1.0018845, 1.0003768, 1.0015694, -0.0012159, 0.0015076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015041, upper bound: 0.0015708
time: 1.17 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014551, upper bound: 0.0015551
time: 1.29 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0049466, 0.0008215, -0.0055233, 0.0057196
1: 0.0032021, 0.0070349, 0.0031430, 0.0072195, -0.0040174, 0.0038919
2: 0.0092501, 0.0195841, 0.0087951, 0.0196588, -0.0092885, 0.0096383
3: -0.0068406, -0.0022609, -0.0068993, -0.0020414, -0.0047991, 0.0046383
4: 0.0038494, 0.0053766, 0.0038301, 0.0054493, -0.0014501, 0.0013979
5: -0.0039186, -0.0004460, -0.0039416, -0.0003068, -0.0036119, 0.0034956
6: -0.0067835, -0.0051295, -0.0068021, -0.0050529, -0.0017306, 0.0016726
7: -0.0036297, -0.0004172, -0.0037825, -0.0003552, -0.0032745, 0.0033653
8: -0.0074341, -0.0005859, -0.0074813, -0.0002877, -0.0071464, 0.0068954
9: 1.0003712, 1.0016996, 1.0003455, 1.0020081, -0.0016369, 0.0013541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016242, upper bound: 0.0016570
time: 1.33 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
time: 1.74 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0049766, 0.0008226, -0.0056699, 0.0057925
1: 0.0031500, 0.0071585, 0.0031411, 0.0072423, -0.0040923, 0.0040174
2: 0.0089850, 0.0196473, 0.0087394, 0.0196612, -0.0094849, 0.0097916
3: -0.0068959, -0.0021200, -0.0068998, -0.0020146, -0.0048813, 0.0047798
4: 0.0038312, 0.0054226, 0.0038299, 0.0054582, -0.0014757, 0.0014145
5: -0.0039354, -0.0003724, -0.0039430, -0.0002896, -0.0036458, 0.0035707
6: -0.0068006, -0.0050825, -0.0068023, -0.0050436, -0.0017570, 0.0017198
7: -0.0037344, -0.0003632, -0.0038014, -0.0003530, -0.0033814, 0.0034382
8: -0.0074731, -0.0004146, -0.0074830, -0.0002510, -0.0072221, 0.0070684
9: 1.0003535, 1.0018845, 1.0003421, 1.0020459, -0.0016924, 0.0015423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016304, upper bound: 0.0016969
time: 2.37 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016290
time: 1.68 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0048378, 0.0006956, -0.0045660, 0.0006846, -0.0055224, 0.0052616
1: 0.0032440, 0.0071585, 0.0032505, 0.0069565, -0.0037125, 0.0039080
2: 0.0090055, 0.0194335, 0.0095071, 0.0194070, -0.0095034, 0.0090351
3: -0.0067841, -0.0021219, -0.0067778, -0.0023629, -0.0044212, 0.0046559
4: 0.0038693, 0.0054216, 0.0038710, 0.0053415, -0.0014257, 0.0014894
5: -0.0038634, -0.0003846, -0.0038510, -0.0005358, -0.0033276, 0.0034664
6: -0.0067606, -0.0050845, -0.0067588, -0.0051695, -0.0015911, 0.0016743
7: -0.0037500, -0.0004389, -0.0035949, -0.0004315, -0.0033185, 0.0031561
8: -0.0073326, -0.0004285, -0.0073140, -0.0007574, -0.0065751, 0.0068855
9: 1.0003529, 1.0018735, 1.0003799, 1.0015326, -0.0011797, 0.0014936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014927, upper bound: 0.0015516
time: 1.24 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014454, upper bound: 0.0015367
time: 1.29 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0045952, 0.0006857, -0.0056540, 0.0053175
1: 0.0032052, 0.0072674, 0.0032486, 0.0069791, -0.0037738, 0.0040188
2: 0.0087682, 0.0194774, 0.0094526, 0.0194093, -0.0096713, 0.0091440
3: -0.0068185, -0.0019956, -0.0067785, -0.0023363, -0.0044822, 0.0047829
4: 0.0038585, 0.0054625, 0.0038708, 0.0053503, -0.0014389, 0.0015019
5: -0.0038752, -0.0003169, -0.0038524, -0.0005188, -0.0033564, 0.0035355
6: -0.0067711, -0.0050429, -0.0067591, -0.0051604, -0.0016107, 0.0017162
7: -0.0038472, -0.0003972, -0.0036138, -0.0004292, -0.0034180, 0.0032165
8: -0.0073620, -0.0002743, -0.0073157, -0.0007215, -0.0066405, 0.0070414
9: 1.0003378, 1.0020391, 1.0003768, 1.0015694, -0.0012316, 0.0016623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015033, upper bound: 0.0015708
time: 1.27 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014546, upper bound: 0.0015551
time: 1.41 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0048378, 0.0006956, -0.0049466, 0.0008215, -0.0056593, 0.0056422
1: 0.0032440, 0.0071585, 0.0031430, 0.0072195, -0.0039755, 0.0040155
2: 0.0090055, 0.0194335, 0.0087951, 0.0196588, -0.0095691, 0.0095377
3: -0.0067841, -0.0021219, -0.0068993, -0.0020414, -0.0047427, 0.0047773
4: 0.0038693, 0.0054216, 0.0038301, 0.0054493, -0.0014661, 0.0014626
5: -0.0038634, -0.0003846, -0.0039416, -0.0003068, -0.0035566, 0.0035570
6: -0.0067606, -0.0050845, -0.0068021, -0.0050529, -0.0017077, 0.0017175
7: -0.0037500, -0.0004389, -0.0037825, -0.0003552, -0.0033947, 0.0033436
8: -0.0073326, -0.0004285, -0.0074813, -0.0002877, -0.0070448, 0.0070528
9: 1.0003529, 1.0018735, 1.0003455, 1.0020081, -0.0016552, 0.0015280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016817, upper bound: 0.0016155
time: 2.08 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016217, upper bound: 0.0016120
time: 1.32 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0049766, 0.0008226, -0.0057909, 0.0056989
1: 0.0032052, 0.0072674, 0.0031411, 0.0072423, -0.0040371, 0.0041263
2: 0.0087682, 0.0194774, 0.0087394, 0.0196612, -0.0097359, 0.0096467
3: -0.0068185, -0.0019956, -0.0068998, -0.0020146, -0.0048039, 0.0049043
4: 0.0038585, 0.0054625, 0.0038299, 0.0054582, -0.0014834, 0.0014776
5: -0.0038752, -0.0003169, -0.0039430, -0.0002896, -0.0035856, 0.0036262
6: -0.0067711, -0.0050429, -0.0068023, -0.0050436, -0.0017275, 0.0017594
7: -0.0038472, -0.0003972, -0.0038014, -0.0003530, -0.0034942, 0.0034042
8: -0.0073620, -0.0002743, -0.0074830, -0.0002510, -0.0071110, 0.0072087
9: 1.0003378, 1.0020391, 1.0003421, 1.0020459, -0.0017080, 0.0016969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016905, upper bound: 0.0016325
time: 1.25 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016269, upper bound: 0.0016290
time: 1.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.80 seconds
IS_B1_A1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0014941, upper bound: 0.0015525
IS_B1_A1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0014464, upper bound: 0.0015370
IS_B1_A1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0015041, upper bound: 0.0015708
IS_B1_A1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0014551, upper bound: 0.0015551
IS_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0016242, upper bound: 0.0016570
IS_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
IS_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0016304, upper bound: 0.0016969
IS_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016290
IS_B1_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0014927, upper bound: 0.0015516
IS_B1_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0014454, upper bound: 0.0015367
IS_B1_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0015033, upper bound: 0.0015708
IS_B1_A2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0014546, upper bound: 0.0015551
IS_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0016817, upper bound: 0.0016155
IS_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0016217, upper bound: 0.0016120
IS_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0016905, upper bound: 0.0016325
IS_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 9, lower bound: -0.0016269, upper bound: 0.0016290

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0049466, 0.0008215, -0.0055135, 0.0055558
1: 0.0033514, 0.0070309, 0.0031430, 0.0072195, -0.0038682, 0.0038879
2: 0.0092692, 0.0192927, 0.0087951, 0.0196588, -0.0092694, 0.0093498
3: -0.0066708, -0.0022665, -0.0068993, -0.0020414, -0.0046294, 0.0046328
4: 0.0039042, 0.0053746, 0.0038301, 0.0054493, -0.0013949, 0.0013958
5: -0.0038431, -0.0004534, -0.0039416, -0.0003068, -0.0035364, 0.0034882
6: -0.0067294, -0.0051322, -0.0068021, -0.0050529, -0.0016764, 0.0016699
7: -0.0036275, -0.0005668, -0.0037825, -0.0003552, -0.0032723, 0.0032157
8: -0.0072464, -0.0005987, -0.0074813, -0.0002877, -0.0069587, 0.0068826
9: 1.0003716, 1.0016876, 1.0003455, 1.0020081, -0.0016365, 0.0013422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
time: 1.34 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
time: 1.35 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0049443, 0.0007928, -0.0058041, 0.0055318
1: 0.0033506, 0.0073251, 0.0031685, 0.0072185, -0.0038679, 0.0041565
2: 0.0087046, 0.0192520, 0.0087996, 0.0196073, -0.0098048, 0.0093510
3: -0.0066545, -0.0019347, -0.0068701, -0.0020428, -0.0046116, 0.0049354
4: 0.0039107, 0.0054817, 0.0038394, 0.0054488, -0.0014267, 0.0015045
5: -0.0038354, -0.0003103, -0.0039273, -0.0003085, -0.0035269, 0.0036171
6: -0.0067229, -0.0050264, -0.0067927, -0.0050536, -0.0016694, 0.0017663
7: -0.0039236, -0.0005583, -0.0037819, -0.0003818, -0.0035418, 0.0032236
8: -0.0072211, -0.0002366, -0.0074474, -0.0002908, -0.0069303, 0.0072109
9: 1.0003281, 1.0020975, 1.0003455, 1.0020052, -0.0016772, 0.0017520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014111, upper bound: 0.0016098
time: 1.53 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014111, upper bound: 0.0016122
time: 1.44 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0049766, 0.0008226, -0.0056596, 0.0056247
1: 0.0033019, 0.0071539, 0.0031411, 0.0072423, -0.0039404, 0.0040129
2: 0.0090055, 0.0193522, 0.0087394, 0.0196612, -0.0094647, 0.0094905
3: -0.0067221, -0.0021260, -0.0068998, -0.0020146, -0.0047076, 0.0047738
4: 0.0038875, 0.0054204, 0.0038299, 0.0054582, -0.0014164, 0.0014121
5: -0.0038581, -0.0003803, -0.0039430, -0.0002896, -0.0035685, 0.0035628
6: -0.0067448, -0.0050854, -0.0068023, -0.0050436, -0.0017013, 0.0017169
7: -0.0037320, -0.0005139, -0.0038014, -0.0003530, -0.0033790, 0.0032875
8: -0.0072838, -0.0004284, -0.0074830, -0.0002510, -0.0070328, 0.0070546
9: 1.0003542, 1.0018717, 1.0003421, 1.0020459, -0.0016917, 0.0015296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016869
time: 1.67 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016969
time: 1.75 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0049743, 0.0007939, -0.0059681, 0.0056134
1: 0.0032942, 0.0074643, 0.0031667, 0.0072413, -0.0039471, 0.0042976
2: 0.0084066, 0.0193341, 0.0087439, 0.0196096, -0.0100387, 0.0095206
3: -0.0067131, -0.0017723, -0.0068707, -0.0020160, -0.0046971, 0.0050983
4: 0.0038907, 0.0055345, 0.0038392, 0.0054577, -0.0014489, 0.0015223
5: -0.0038575, -0.0002268, -0.0039288, -0.0002913, -0.0035662, 0.0037020
6: -0.0067417, -0.0049735, -0.0067930, -0.0050442, -0.0016975, 0.0018195
7: -0.0040409, -0.0005022, -0.0038008, -0.0003795, -0.0036614, 0.0032986
8: -0.0072705, -0.0000416, -0.0074491, -0.0002541, -0.0070164, 0.0074075
9: 1.0003084, 1.0023073, 1.0003424, 1.0020431, -0.0017347, 0.0019649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016228
time: 1.38 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016284
time: 1.52 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0048378, 0.0006956, -0.0049361, 0.0006538, -0.0054915, 0.0056318
1: 0.0032440, 0.0071585, 0.0032947, 0.0072150, -0.0039710, 0.0038637
2: 0.0090055, 0.0194335, 0.0088157, 0.0193639, -0.0092680, 0.0095173
3: -0.0067841, -0.0021219, -0.0067254, -0.0020476, -0.0047365, 0.0046035
4: 0.0038693, 0.0054216, 0.0038864, 0.0054472, -0.0014638, 0.0014033
5: -0.0038634, -0.0003846, -0.0038643, -0.0003146, -0.0035488, 0.0034797
6: -0.0067606, -0.0050845, -0.0067463, -0.0050558, -0.0017048, 0.0016618
7: -0.0037500, -0.0004389, -0.0037799, -0.0005057, -0.0032442, 0.0033410
8: -0.0073326, -0.0004285, -0.0072921, -0.0003016, -0.0070310, 0.0068637
9: 1.0003529, 1.0018735, 1.0003459, 1.0019951, -0.0016422, 0.0015275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016217, upper bound: 0.0016120
time: 1.90 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016217, upper bound: 0.0016120
time: 1.77 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048353, 0.0006653, -0.0052674, 0.0006450, -0.0054803, 0.0059327
1: 0.0032701, 0.0071575, 0.0032858, 0.0075192, -0.0042491, 0.0038716
2: 0.0090103, 0.0193793, 0.0082278, 0.0193459, -0.0092978, 0.0100850
3: -0.0067545, -0.0021233, -0.0067167, -0.0017042, -0.0050502, 0.0045934
4: 0.0038790, 0.0054212, 0.0038895, 0.0055577, -0.0015771, 0.0014358
5: -0.0038486, -0.0003865, -0.0038643, -0.0001632, -0.0036854, 0.0034778
6: -0.0067507, -0.0050852, -0.0067432, -0.0049468, -0.0018040, 0.0016580
7: -0.0037494, -0.0004657, -0.0040840, -0.0004947, -0.0032548, 0.0036183
8: -0.0072976, -0.0004317, -0.0072786, 0.0000777, -0.0073752, 0.0068469
9: 1.0003531, 1.0018706, 1.0003011, 1.0024213, -0.0020682, 0.0015695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016095
time: 1.80 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016120
time: 1.51 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0049661, 0.0006549, -0.0056232, 0.0056884
1: 0.0032052, 0.0072674, 0.0032928, 0.0072377, -0.0040325, 0.0039746
2: 0.0087682, 0.0194774, 0.0087600, 0.0193663, -0.0094381, 0.0096261
3: -0.0068185, -0.0019956, -0.0067260, -0.0020208, -0.0047977, 0.0047305
4: 0.0038585, 0.0054625, 0.0038862, 0.0054561, -0.0014811, 0.0014215
5: -0.0038752, -0.0003169, -0.0038657, -0.0002975, -0.0035777, 0.0035489
6: -0.0067711, -0.0050429, -0.0067466, -0.0050465, -0.0017246, 0.0017037
7: -0.0038472, -0.0003972, -0.0037987, -0.0005034, -0.0033438, 0.0034015
8: -0.0073620, -0.0002743, -0.0072939, -0.0002649, -0.0070971, 0.0070196
9: 1.0003378, 1.0020391, 1.0003427, 1.0020331, -0.0016953, 0.0016963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016252
time: 1.91 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016313
time: 1.51 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049659, 0.0006930, -0.0052987, 0.0006461, -0.0056119, 0.0059917
1: 0.0032308, 0.0072664, 0.0032829, 0.0075432, -0.0043124, 0.0039835
2: 0.0087729, 0.0194241, 0.0081705, 0.0193481, -0.0094647, 0.0101977
3: -0.0067892, -0.0019970, -0.0067173, -0.0016758, -0.0051134, 0.0047204
4: 0.0038680, 0.0054620, 0.0038893, 0.0055671, -0.0015959, 0.0014551
5: -0.0038607, -0.0003187, -0.0038662, -0.0001453, -0.0037153, 0.0035475
6: -0.0067614, -0.0050436, -0.0067435, -0.0049370, -0.0018245, 0.0016999
7: -0.0038466, -0.0004232, -0.0041039, -0.0004923, -0.0033543, 0.0036807
8: -0.0073273, -0.0002775, -0.0072803, 0.0001155, -0.0074428, 0.0070028
9: 1.0003380, 1.0020361, 1.0002978, 1.0024607, -0.0021228, 0.0017383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016228
time: 1.69 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016284
time: 1.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.67 seconds
IS_B1_A1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
IS_B1_A1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
IS_B1_A1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0014111, upper bound: 0.0016098
IS_B1_A1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0014111, upper bound: 0.0016122
IS_B1_A1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016869
IS_B1_A1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016969
IS_B1_A1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016228
IS_B1_A1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016284
IS_B1_A2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016217, upper bound: 0.0016120
IS_B1_A2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016217, upper bound: 0.0016120
IS_B1_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016095
IS_B1_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016120
IS_B1_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016252
IS_B1_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016313
IS_B1_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016228
IS_B1_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016284

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0049361, 0.0006538, -0.0053458, 0.0055453
1: 0.0033514, 0.0070309, 0.0032947, 0.0072150, -0.0038637, 0.0037362
2: 0.0092692, 0.0192927, 0.0088157, 0.0193639, -0.0089683, 0.0093294
3: -0.0066708, -0.0022665, -0.0067254, -0.0020476, -0.0046232, 0.0044590
4: 0.0039042, 0.0053746, 0.0038864, 0.0054472, -0.0013925, 0.0013365
5: -0.0038431, -0.0004534, -0.0038643, -0.0003146, -0.0035285, 0.0034109
6: -0.0067294, -0.0051322, -0.0067463, -0.0050558, -0.0016735, 0.0016141
7: -0.0036275, -0.0005668, -0.0037799, -0.0005057, -0.0031218, 0.0032131
8: -0.0072464, -0.0005987, -0.0072921, -0.0003016, -0.0069449, 0.0066935
9: 1.0003716, 1.0016876, 1.0003459, 1.0019951, -0.0016235, 0.0013417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016555
time: 1.26 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016570
time: 1.34 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0052674, 0.0006450, -0.0053370, 0.0058766
1: 0.0033514, 0.0070309, 0.0032858, 0.0075192, -0.0041678, 0.0037450
2: 0.0092692, 0.0192927, 0.0082278, 0.0193459, -0.0089810, 0.0099491
3: -0.0066708, -0.0022665, -0.0067167, -0.0017042, -0.0049666, 0.0044502
4: 0.0039042, 0.0053746, 0.0038895, 0.0055577, -0.0015155, 0.0013520
5: -0.0038431, -0.0004534, -0.0038643, -0.0001632, -0.0036799, 0.0034109
6: -0.0067294, -0.0051322, -0.0067432, -0.0049468, -0.0017826, 0.0016110
7: -0.0036275, -0.0005668, -0.0040840, -0.0004947, -0.0031329, 0.0035172
8: -0.0072464, -0.0005987, -0.0072786, 0.0000777, -0.0073241, 0.0066800
9: 1.0003716, 1.0016876, 1.0003011, 1.0024213, -0.0020497, 0.0013865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016555
time: 1.52 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016570
time: 1.77 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0046994, 0.0007435, -0.0057548, 0.0052869
1: 0.0033506, 0.0073251, 0.0032282, 0.0070339, -0.0036833, 0.0040969
2: 0.0087046, 0.0192520, 0.0092547, 0.0195307, -0.0096923, 0.0088932
3: -0.0066545, -0.0019347, -0.0068101, -0.0022622, -0.0043922, 0.0048754
4: 0.0039107, 0.0054817, 0.0038592, 0.0053761, -0.0013563, 0.0014856
5: -0.0038354, -0.0003103, -0.0039039, -0.0004478, -0.0033876, 0.0035936
6: -0.0067229, -0.0050264, -0.0067738, -0.0051301, -0.0015928, 0.0017474
7: -0.0039236, -0.0005583, -0.0036292, -0.0004436, -0.0034800, 0.0030709
8: -0.0072211, -0.0002366, -0.0073994, -0.0005889, -0.0066321, 0.0071629
9: 1.0003281, 1.0020975, 1.0003713, 1.0016966, -0.0013685, 0.0017262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
time: 2.01 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
time: 1.42 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0048450, 0.0007871, -0.0057984, 0.0054325
1: 0.0033506, 0.0073251, 0.0031755, 0.0071574, -0.0038068, 0.0041496
2: 0.0087046, 0.0192520, 0.0089895, 0.0195958, -0.0097920, 0.0091779
3: -0.0066545, -0.0019347, -0.0068668, -0.0021214, -0.0045331, 0.0049321
4: 0.0039107, 0.0054817, 0.0038405, 0.0054221, -0.0014009, 0.0015029
5: -0.0038354, -0.0003103, -0.0039210, -0.0003741, -0.0034613, 0.0036108
6: -0.0067229, -0.0050264, -0.0067912, -0.0050832, -0.0016397, 0.0017648
7: -0.0039236, -0.0005583, -0.0037339, -0.0003896, -0.0035340, 0.0031755
8: -0.0072211, -0.0002366, -0.0074392, -0.0004176, -0.0068034, 0.0072027
9: 1.0003281, 1.0020975, 1.0003538, 1.0018815, -0.0015534, 0.0017437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016122
time: 1.87 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
time: 2.32 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0047018, 0.0007730, -0.0056100, 0.0053498
1: 0.0033019, 0.0071539, 0.0032021, 0.0070349, -0.0037331, 0.0039518
2: 0.0090055, 0.0193522, 0.0092501, 0.0195841, -0.0094448, 0.0089745
3: -0.0067221, -0.0021260, -0.0068406, -0.0022609, -0.0044612, 0.0047146
4: 0.0038875, 0.0054204, 0.0038494, 0.0053766, -0.0013371, 0.0014220
5: -0.0038581, -0.0003803, -0.0039186, -0.0004460, -0.0034121, 0.0035384
6: -0.0067448, -0.0050854, -0.0067835, -0.0051295, -0.0016154, 0.0016981
7: -0.0037320, -0.0005139, -0.0036297, -0.0004172, -0.0033148, 0.0031158
8: -0.0072838, -0.0004284, -0.0074341, -0.0005859, -0.0066980, 0.0070057
9: 1.0003542, 1.0018717, 1.0003712, 1.0016996, -0.0013454, 0.0015005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016869
time: 1.84 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016869
time: 1.68 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0048472, 0.0008159, -0.0056528, 0.0054953
1: 0.0033019, 0.0071539, 0.0031500, 0.0071585, -0.0038566, 0.0040040
2: 0.0090055, 0.0193522, 0.0089850, 0.0196473, -0.0094490, 0.0091712
3: -0.0067221, -0.0021260, -0.0068959, -0.0021200, -0.0046021, 0.0047699
4: 0.0038875, 0.0054204, 0.0038312, 0.0054226, -0.0013565, 0.0014102
5: -0.0038581, -0.0003803, -0.0039354, -0.0003724, -0.0034858, 0.0035551
6: -0.0067448, -0.0050854, -0.0068006, -0.0050825, -0.0016623, 0.0017152
7: -0.0037320, -0.0005139, -0.0037344, -0.0003632, -0.0033688, 0.0032205
8: -0.0072838, -0.0004284, -0.0074731, -0.0004146, -0.0068693, 0.0070447
9: 1.0003542, 1.0018717, 1.0003535, 1.0018845, -0.0015303, 0.0015182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016969
time: 2.17 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016969
time: 1.78 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0046994, 0.0007435, -0.0059177, 0.0053386
1: 0.0032942, 0.0074643, 0.0032282, 0.0070339, -0.0037398, 0.0042362
2: 0.0084066, 0.0193341, 0.0092547, 0.0195307, -0.0100237, 0.0090047
3: -0.0067131, -0.0017723, -0.0068101, -0.0022622, -0.0044508, 0.0050377
4: 0.0038907, 0.0055345, 0.0038592, 0.0053761, -0.0013696, 0.0015380
5: -0.0038575, -0.0002268, -0.0039039, -0.0004478, -0.0034097, 0.0036771
6: -0.0067417, -0.0049735, -0.0067738, -0.0051301, -0.0016116, 0.0018003
7: -0.0040409, -0.0005022, -0.0036292, -0.0004436, -0.0035973, 0.0031270
8: -0.0072705, -0.0000416, -0.0073994, -0.0005889, -0.0066815, 0.0073579
9: 1.0003084, 1.0023073, 1.0003713, 1.0016966, -0.0013882, 0.0019360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016228
time: 1.44 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016228
time: 1.33 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0048450, 0.0007871, -0.0059613, 0.0054841
1: 0.0032942, 0.0074643, 0.0031755, 0.0071574, -0.0038633, 0.0042888
2: 0.0084066, 0.0193341, 0.0089895, 0.0195958, -0.0100231, 0.0091987
3: -0.0067131, -0.0017723, -0.0068668, -0.0021214, -0.0045917, 0.0050944
4: 0.0038907, 0.0055345, 0.0038405, 0.0054221, -0.0013901, 0.0015204
5: -0.0038575, -0.0002268, -0.0039210, -0.0003741, -0.0034834, 0.0036942
6: -0.0067417, -0.0049735, -0.0067912, -0.0050832, -0.0016585, 0.0018177
7: -0.0040409, -0.0005022, -0.0037339, -0.0003896, -0.0036513, 0.0032317
8: -0.0072705, -0.0000416, -0.0074392, -0.0004176, -0.0068528, 0.0073977
9: 1.0003084, 1.0023073, 1.0003538, 1.0018815, -0.0015731, 0.0019535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016284
time: 1.78 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016284
time: 1.47 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0048278, 0.0005301, -0.0049361, 0.0006538, -0.0054815, 0.0054662
1: 0.0033941, 0.0071543, 0.0032947, 0.0072150, -0.0038209, 0.0038596
2: 0.0090253, 0.0191401, 0.0088157, 0.0193639, -0.0092476, 0.0092214
3: -0.0066109, -0.0021276, -0.0067254, -0.0020476, -0.0045633, 0.0045978
4: 0.0039253, 0.0054196, 0.0038864, 0.0054472, -0.0014091, 0.0014009
5: -0.0037876, -0.0003922, -0.0038643, -0.0003146, -0.0034730, 0.0034721
6: -0.0067061, -0.0050873, -0.0067463, -0.0050558, -0.0016503, 0.0016590
7: -0.0037476, -0.0005895, -0.0037799, -0.0005057, -0.0032419, 0.0031904
8: -0.0071428, -0.0004418, -0.0072921, -0.0003016, -0.0068413, 0.0068504
9: 1.0003535, 1.0018612, 1.0003459, 1.0019951, -0.0016416, 0.0015153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016526, upper bound: 0.0016122
time: 1.50 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016526, upper bound: 0.0016154
time: 1.60 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0051470, 0.0005027, -0.0049361, 0.0006538, -0.0058008, 0.0054388
1: 0.0033933, 0.0074547, 0.0032947, 0.0072150, -0.0038217, 0.0041600
2: 0.0084569, 0.0190914, 0.0088157, 0.0193639, -0.0098387, 0.0091963
3: -0.0065925, -0.0017877, -0.0067254, -0.0020476, -0.0045449, 0.0049377
4: 0.0039324, 0.0055289, 0.0038864, 0.0054472, -0.0014056, 0.0015118
5: -0.0037792, -0.0002434, -0.0038643, -0.0003146, -0.0034646, 0.0036209
6: -0.0066975, -0.0049808, -0.0067463, -0.0050558, -0.0016417, 0.0017655
7: -0.0040536, -0.0005844, -0.0037799, -0.0005057, -0.0035479, 0.0031955
8: -0.0071131, -0.0000742, -0.0072921, -0.0003016, -0.0068115, 0.0072179
9: 1.0003088, 1.0022757, 1.0003459, 1.0019951, -0.0016863, 0.0019298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016526, upper bound: 0.0016122
time: 1.99 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016526, upper bound: 0.0016155
time: 1.74 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0048353, 0.0006653, -0.0050113, 0.0005875, -0.0054228, 0.0056766
1: 0.0032701, 0.0071575, 0.0033506, 0.0073251, -0.0040550, 0.0038068
2: 0.0090103, 0.0193793, 0.0087046, 0.0192520, -0.0091734, 0.0095926
3: -0.0067545, -0.0021233, -0.0066545, -0.0019347, -0.0048198, 0.0045312
4: 0.0038790, 0.0054212, 0.0039107, 0.0054817, -0.0015008, 0.0014209
5: -0.0038486, -0.0003865, -0.0038354, -0.0003103, -0.0035383, 0.0034489
6: -0.0067507, -0.0050852, -0.0067229, -0.0050264, -0.0017243, 0.0016377
7: -0.0037494, -0.0004657, -0.0039236, -0.0005583, -0.0031911, 0.0034579
8: -0.0072976, -0.0004317, -0.0072211, -0.0002366, -0.0070610, 0.0067894
9: 1.0003531, 1.0018706, 1.0003281, 1.0020975, -0.0017444, 0.0015426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016095
time: 1.38 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016095
time: 1.94 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0048353, 0.0006653, -0.0051742, 0.0006392, -0.0054745, 0.0058395
1: 0.0032701, 0.0071575, 0.0032942, 0.0074643, -0.0041943, 0.0038633
2: 0.0090103, 0.0193793, 0.0084066, 0.0193341, -0.0092849, 0.0099240
3: -0.0067545, -0.0021233, -0.0067131, -0.0017723, -0.0049821, 0.0045898
4: 0.0038790, 0.0054212, 0.0038907, 0.0055345, -0.0015532, 0.0014341
5: -0.0038486, -0.0003865, -0.0038575, -0.0002268, -0.0036218, 0.0034710
6: -0.0067507, -0.0050852, -0.0067417, -0.0049735, -0.0017772, 0.0016565
7: -0.0037494, -0.0004657, -0.0040409, -0.0005022, -0.0032472, 0.0035752
8: -0.0072976, -0.0004317, -0.0072705, -0.0000416, -0.0072560, 0.0068387
9: 1.0003531, 1.0018706, 1.0003084, 1.0023073, -0.0019542, 0.0015622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016120
time: 1.72 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016120
time: 1.74 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0046920, 0.0006092, -0.0055774, 0.0054143
1: 0.0032052, 0.0072674, 0.0033514, 0.0070309, -0.0038257, 0.0039160
2: 0.0087682, 0.0194774, 0.0092692, 0.0192927, -0.0094154, 0.0091118
3: -0.0068185, -0.0019956, -0.0066708, -0.0022665, -0.0045520, 0.0046753
4: 0.0038585, 0.0054625, 0.0039042, 0.0053746, -0.0014019, 0.0014257
5: -0.0038752, -0.0003169, -0.0038431, -0.0004534, -0.0034218, 0.0035263
6: -0.0067711, -0.0050429, -0.0067294, -0.0051322, -0.0016389, 0.0016865
7: -0.0038472, -0.0003972, -0.0036275, -0.0005668, -0.0032804, 0.0032303
8: -0.0073620, -0.0002743, -0.0072464, -0.0005987, -0.0067633, 0.0069721
9: 1.0003378, 1.0020391, 1.0003716, 1.0016876, -0.0013498, 0.0016675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016252
time: 2.02 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016252
time: 2.13 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0048370, 0.0006481, -0.0056164, 0.0055593
1: 0.0032052, 0.0072674, 0.0033019, 0.0071539, -0.0039487, 0.0039655
2: 0.0087682, 0.0194774, 0.0090055, 0.0193522, -0.0094221, 0.0093181
3: -0.0068185, -0.0019956, -0.0067221, -0.0021260, -0.0046925, 0.0047266
4: 0.0038585, 0.0054625, 0.0038875, 0.0054204, -0.0014251, 0.0014196
5: -0.0038752, -0.0003169, -0.0038581, -0.0003803, -0.0034949, 0.0035413
6: -0.0067711, -0.0050429, -0.0067448, -0.0050854, -0.0016857, 0.0017019
7: -0.0038472, -0.0003972, -0.0037320, -0.0005139, -0.0033333, 0.0033348
8: -0.0073620, -0.0002743, -0.0072838, -0.0004284, -0.0069336, 0.0070095
9: 1.0003378, 1.0020391, 1.0003542, 1.0018717, -0.0015339, 0.0016849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016313
time: 1.62 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016313
time: 1.48 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0049659, 0.0006930, -0.0050113, 0.0005875, -0.0055533, 0.0057043
1: 0.0032308, 0.0072664, 0.0033506, 0.0073251, -0.0040943, 0.0039157
2: 0.0087729, 0.0194241, 0.0087046, 0.0192520, -0.0094164, 0.0096450
3: -0.0067892, -0.0019970, -0.0066545, -0.0019347, -0.0048545, 0.0046575
4: 0.0038680, 0.0054620, 0.0039107, 0.0054817, -0.0015100, 0.0014575
5: -0.0038607, -0.0003187, -0.0038354, -0.0003103, -0.0035504, 0.0035167
6: -0.0067614, -0.0050436, -0.0067229, -0.0050264, -0.0017350, 0.0016794
7: -0.0038466, -0.0004232, -0.0039236, -0.0005583, -0.0032883, 0.0035004
8: -0.0073273, -0.0002775, -0.0072211, -0.0002366, -0.0070908, 0.0069436
9: 1.0003380, 1.0020361, 1.0003281, 1.0020975, -0.0017595, 0.0017080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016228
time: 1.97 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016228
time: 2.07 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0049659, 0.0006930, -0.0051742, 0.0006392, -0.0056050, 0.0058672
1: 0.0032308, 0.0072664, 0.0032942, 0.0074643, -0.0042335, 0.0039722
2: 0.0087729, 0.0194241, 0.0084066, 0.0193341, -0.0094493, 0.0098919
3: -0.0067892, -0.0019970, -0.0067131, -0.0017723, -0.0050169, 0.0047161
4: 0.0038680, 0.0054620, 0.0038907, 0.0055345, -0.0015345, 0.0014531
5: -0.0038607, -0.0003187, -0.0038575, -0.0002268, -0.0036339, 0.0035388
6: -0.0067614, -0.0050436, -0.0067417, -0.0049735, -0.0017879, 0.0016981
7: -0.0038466, -0.0004232, -0.0040409, -0.0005022, -0.0033444, 0.0036177
8: -0.0073273, -0.0002775, -0.0072705, -0.0000416, -0.0072858, 0.0069930
9: 1.0003380, 1.0020361, 1.0003084, 1.0023073, -0.0019693, 0.0017277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016284
time: 1.38 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016284
time: 1.32 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.92 seconds
IS_B1_A1_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016555
IS_B1_A1_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016570
IS_B1_A1_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016555
IS_B1_A1_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016570
IS_B1_A1_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
IS_B1_A1_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
IS_B1_A1_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016122
IS_B1_A1_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
IS_B1_A1_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016869
IS_B1_A1_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016869
IS_B1_A1_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016969
IS_B1_A1_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016969
IS_B1_A1_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016228
IS_B1_A1_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016228
IS_B1_A1_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016284
IS_B1_A1_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016284
IS_B1_A2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016526, upper bound: 0.0016122
IS_B1_A2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016526, upper bound: 0.0016154
IS_B1_A2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016526, upper bound: 0.0016122
IS_B1_A2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016526, upper bound: 0.0016155
IS_B1_A2_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016095
IS_B1_A2_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016095
IS_B1_A2_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016120
IS_B1_A2_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016086, upper bound: 0.0016120
IS_B1_A2_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016252
IS_B1_A2_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016252
IS_B1_A2_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016313
IS_B1_A2_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016535, upper bound: 0.0016313
IS_B1_A2_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016228
IS_B1_A2_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016228
IS_B1_A2_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016284
IS_B1_A2_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 9, lower bound: -0.0016103, upper bound: 0.0016284

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0046920, 0.0006092, -0.0053012, 0.0053012
1: 0.0033514, 0.0070309, 0.0033514, 0.0070309, -0.0036795, 0.0036795
2: 0.0092692, 0.0192927, 0.0092692, 0.0192927, -0.0088730, 0.0088730
3: -0.0066708, -0.0022665, -0.0066708, -0.0022665, -0.0044044, 0.0044044
4: 0.0039042, 0.0053746, 0.0039042, 0.0053746, -0.0013223, 0.0013223
5: -0.0038431, -0.0004534, -0.0038431, -0.0004534, -0.0033898, 0.0033898
6: -0.0067294, -0.0051322, -0.0067294, -0.0051322, -0.0015972, 0.0015972
7: -0.0036275, -0.0005668, -0.0036275, -0.0005668, -0.0030608, 0.0030608
8: -0.0072464, -0.0005987, -0.0072464, -0.0005987, -0.0066477, 0.0066477
9: 1.0003716, 1.0016876, 1.0003716, 1.0016876, -0.0013161, 0.0013161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016015, upper bound: 0.0015349
time: 1.30 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016552, upper bound: 0.0016569
time: 1.51 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0048370, 0.0006481, -0.0053401, 0.0054461
1: 0.0033514, 0.0070309, 0.0033019, 0.0071539, -0.0038026, 0.0037290
2: 0.0092692, 0.0192927, 0.0090055, 0.0193522, -0.0089555, 0.0091564
3: -0.0066708, -0.0022665, -0.0067221, -0.0021260, -0.0045448, 0.0044557
4: 0.0039042, 0.0053746, 0.0038875, 0.0054204, -0.0013668, 0.0013349
5: -0.0038431, -0.0004534, -0.0038581, -0.0003803, -0.0034629, 0.0034048
6: -0.0067294, -0.0051322, -0.0067448, -0.0050854, -0.0016440, 0.0016127
7: -0.0036275, -0.0005668, -0.0037320, -0.0005139, -0.0031136, 0.0031652
8: -0.0072464, -0.0005987, -0.0072838, -0.0004284, -0.0068180, 0.0066852
9: 1.0003716, 1.0016876, 1.0003542, 1.0018717, -0.0015001, 0.0013335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016119
time: 1.26 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016552, upper bound: 0.0016601
time: 1.69 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0050113, 0.0005875, -0.0052795, 0.0056205
1: 0.0033514, 0.0070309, 0.0033506, 0.0073251, -0.0039737, 0.0036802
2: 0.0092692, 0.0192927, 0.0087046, 0.0192520, -0.0088492, 0.0094568
3: -0.0066708, -0.0022665, -0.0066545, -0.0019347, -0.0047361, 0.0043880
4: 0.0039042, 0.0053746, 0.0039107, 0.0054817, -0.0014391, 0.0013284
5: -0.0038431, -0.0004534, -0.0038354, -0.0003103, -0.0035329, 0.0033820
6: -0.0067294, -0.0051322, -0.0067229, -0.0050264, -0.0017030, 0.0015908
7: -0.0036275, -0.0005668, -0.0039236, -0.0005583, -0.0030692, 0.0033568
8: -0.0072464, -0.0005987, -0.0072211, -0.0002366, -0.0070098, 0.0066224
9: 1.0003716, 1.0016876, 1.0003281, 1.0020975, -0.0017259, 0.0013596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014682, upper bound: 0.0015568
time: 1.37 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015987, upper bound: 0.0016414
time: 1.35 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0051742, 0.0006392, -0.0053312, 0.0057833
1: 0.0033514, 0.0070309, 0.0032942, 0.0074643, -0.0041130, 0.0037367
2: 0.0092692, 0.0192927, 0.0084066, 0.0193341, -0.0089681, 0.0097881
3: -0.0066708, -0.0022665, -0.0067131, -0.0017723, -0.0048985, 0.0044466
4: 0.0039042, 0.0053746, 0.0038907, 0.0055345, -0.0014915, 0.0013503
5: -0.0038431, -0.0004534, -0.0038575, -0.0002268, -0.0036163, 0.0034041
6: -0.0067294, -0.0051322, -0.0067417, -0.0049735, -0.0017559, 0.0016095
7: -0.0036275, -0.0005668, -0.0040409, -0.0005022, -0.0031253, 0.0034741
8: -0.0072464, -0.0005987, -0.0072705, -0.0000416, -0.0072048, 0.0066718
9: 1.0003716, 1.0016876, 1.0003084, 1.0023073, -0.0019357, 0.0013793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014682, upper bound: 0.0015596
time: 1.39 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015987, upper bound: 0.0016433
time: 1.32 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0046920, 0.0006092, -0.0056205, 0.0052795
1: 0.0033506, 0.0073251, 0.0033514, 0.0070309, -0.0036802, 0.0039737
2: 0.0087046, 0.0192520, 0.0092692, 0.0192927, -0.0094568, 0.0088492
3: -0.0066545, -0.0019347, -0.0066708, -0.0022665, -0.0043880, 0.0047361
4: 0.0039107, 0.0054817, 0.0039042, 0.0053746, -0.0013284, 0.0014391
5: -0.0038354, -0.0003103, -0.0038431, -0.0004534, -0.0033820, 0.0035329
6: -0.0067229, -0.0050264, -0.0067294, -0.0051322, -0.0015908, 0.0017030
7: -0.0039236, -0.0005583, -0.0036275, -0.0005668, -0.0033568, 0.0030692
8: -0.0072211, -0.0002366, -0.0072464, -0.0005987, -0.0066224, 0.0070098
9: 1.0003281, 1.0020975, 1.0003716, 1.0016876, -0.0013596, 0.0017259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015187, upper bound: 0.0014621
time: 3.02 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015958, upper bound: 0.0015962
time: 1.34 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0050113, 0.0005875, -0.0055988, 0.0055988
1: 0.0033506, 0.0073251, 0.0033506, 0.0073251, -0.0039744, 0.0039744
2: 0.0087046, 0.0192520, 0.0087046, 0.0192520, -0.0093273, 0.0093273
3: -0.0066545, -0.0019347, -0.0066545, -0.0019347, -0.0047198, 0.0047198
4: 0.0039107, 0.0054817, 0.0039107, 0.0054817, -0.0013711, 0.0013711
5: -0.0038354, -0.0003103, -0.0038354, -0.0003103, -0.0035251, 0.0035251
6: -0.0067229, -0.0050264, -0.0067229, -0.0050264, -0.0016965, 0.0016965
7: -0.0039236, -0.0005583, -0.0039236, -0.0005583, -0.0033653, 0.0033653
8: -0.0072211, -0.0002366, -0.0072211, -0.0002366, -0.0069845, 0.0069845
9: 1.0003281, 1.0020975, 1.0003281, 1.0020975, -0.0017694, 0.0017694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015187, upper bound: 0.0014621
time: 1.41 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015958, upper bound: 0.0015962
time: 1.47 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0048370, 0.0006481, -0.0056594, 0.0054245
1: 0.0033506, 0.0073251, 0.0033019, 0.0071539, -0.0038033, 0.0040232
2: 0.0087046, 0.0192520, 0.0090055, 0.0193522, -0.0095393, 0.0091325
3: -0.0066545, -0.0019347, -0.0067221, -0.0021260, -0.0045285, 0.0047874
4: 0.0039107, 0.0054817, 0.0038875, 0.0054204, -0.0013729, 0.0014517
5: -0.0038354, -0.0003103, -0.0038581, -0.0003803, -0.0034551, 0.0035479
6: -0.0067229, -0.0050264, -0.0067448, -0.0050854, -0.0016375, 0.0017184
7: -0.0039236, -0.0005583, -0.0037320, -0.0005139, -0.0034097, 0.0031737
8: -0.0072211, -0.0002366, -0.0072838, -0.0004284, -0.0067927, 0.0070473
9: 1.0003281, 1.0020975, 1.0003542, 1.0018717, -0.0015436, 0.0017433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0014659
time: 2.34 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015987
time: 1.60 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0051742, 0.0006392, -0.0056504, 0.0057616
1: 0.0033506, 0.0073251, 0.0032942, 0.0074643, -0.0041137, 0.0040309
2: 0.0087046, 0.0192520, 0.0084066, 0.0193341, -0.0094388, 0.0096547
3: -0.0066545, -0.0019347, -0.0067131, -0.0017723, -0.0048821, 0.0047783
4: 0.0039107, 0.0054817, 0.0038907, 0.0055345, -0.0014185, 0.0013843
5: -0.0038354, -0.0003103, -0.0038575, -0.0002268, -0.0036086, 0.0035473
6: -0.0067229, -0.0050264, -0.0067417, -0.0049735, -0.0017494, 0.0017153
7: -0.0039236, -0.0005583, -0.0040409, -0.0005022, -0.0034214, 0.0034826
8: -0.0072211, -0.0002366, -0.0072705, -0.0000416, -0.0071795, 0.0070339
9: 1.0003281, 1.0020975, 1.0003084, 1.0023073, -0.0019792, 0.0017891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014674, upper bound: 0.0015253
time: 1.20 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015987
time: 1.26 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0046920, 0.0006092, -0.0054461, 0.0053401
1: 0.0033019, 0.0071539, 0.0033514, 0.0070309, -0.0037290, 0.0038026
2: 0.0090055, 0.0193522, 0.0092692, 0.0192927, -0.0091564, 0.0089555
3: -0.0067221, -0.0021260, -0.0066708, -0.0022665, -0.0044557, 0.0045448
4: 0.0038875, 0.0054204, 0.0039042, 0.0053746, -0.0013349, 0.0013668
5: -0.0038581, -0.0003803, -0.0038431, -0.0004534, -0.0034048, 0.0034629
6: -0.0067448, -0.0050854, -0.0067294, -0.0051322, -0.0016127, 0.0016440
7: -0.0037320, -0.0005139, -0.0036275, -0.0005668, -0.0031652, 0.0031136
8: -0.0072838, -0.0004284, -0.0072464, -0.0005987, -0.0066852, 0.0068180
9: 1.0003542, 1.0018717, 1.0003716, 1.0016876, -0.0013335, 0.0015001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
time: 1.25 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016013, upper bound: 0.0016726
time: 1.68 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0050113, 0.0005875, -0.0054245, 0.0056594
1: 0.0033019, 0.0071539, 0.0033506, 0.0073251, -0.0040232, 0.0038033
2: 0.0090055, 0.0193522, 0.0087046, 0.0192520, -0.0091325, 0.0095393
3: -0.0067221, -0.0021260, -0.0066545, -0.0019347, -0.0047874, 0.0045285
4: 0.0038875, 0.0054204, 0.0039107, 0.0054817, -0.0014517, 0.0013729
5: -0.0038581, -0.0003803, -0.0038354, -0.0003103, -0.0035479, 0.0034551
6: -0.0067448, -0.0050854, -0.0067229, -0.0050264, -0.0017184, 0.0016375
7: -0.0037320, -0.0005139, -0.0039236, -0.0005583, -0.0031737, 0.0034097
8: -0.0072838, -0.0004284, -0.0072211, -0.0002366, -0.0070473, 0.0067927
9: 1.0003542, 1.0018717, 1.0003281, 1.0020975, -0.0017433, 0.0015436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0015775
time: 1.71 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016013, upper bound: 0.0016726
time: 1.53 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0048370, 0.0006481, -0.0054851, 0.0054851
1: 0.0033019, 0.0071539, 0.0033019, 0.0071539, -0.0038521, 0.0038521
2: 0.0090055, 0.0193522, 0.0090055, 0.0193522, -0.0091510, 0.0091510
3: -0.0067221, -0.0021260, -0.0067221, -0.0021260, -0.0045961, 0.0045961
4: 0.0038875, 0.0054204, 0.0038875, 0.0054204, -0.0013542, 0.0013542
5: -0.0038581, -0.0003803, -0.0038581, -0.0003803, -0.0034779, 0.0034779
6: -0.0067448, -0.0050854, -0.0067448, -0.0050854, -0.0016594, 0.0016594
7: -0.0037320, -0.0005139, -0.0037320, -0.0005139, -0.0032181, 0.0032181
8: -0.0072838, -0.0004284, -0.0072838, -0.0004284, -0.0068555, 0.0068555
9: 1.0003542, 1.0018717, 1.0003542, 1.0018717, -0.0015175, 0.0015175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0015418
time: 1.31 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016829
time: 1.47 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0051742, 0.0006392, -0.0054761, 0.0058222
1: 0.0033019, 0.0071539, 0.0032942, 0.0074643, -0.0041625, 0.0038598
2: 0.0090055, 0.0193522, 0.0084066, 0.0193341, -0.0091548, 0.0097747
3: -0.0067221, -0.0021260, -0.0067131, -0.0017723, -0.0049498, 0.0045871
4: 0.0038875, 0.0054204, 0.0038907, 0.0055345, -0.0014731, 0.0013617
5: -0.0038581, -0.0003803, -0.0038575, -0.0002268, -0.0036313, 0.0034772
6: -0.0067448, -0.0050854, -0.0067417, -0.0049735, -0.0017714, 0.0016563
7: -0.0037320, -0.0005139, -0.0040409, -0.0005022, -0.0032298, 0.0035270
8: -0.0072838, -0.0004284, -0.0072705, -0.0000416, -0.0072423, 0.0068421
9: 1.0003542, 1.0018717, 1.0003084, 1.0023073, -0.0019531, 0.0015633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014823, upper bound: 0.0015904
time: 1.39 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016829
time: 1.27 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0046920, 0.0006092, -0.0057833, 0.0053312
1: 0.0032942, 0.0074643, 0.0033514, 0.0070309, -0.0037367, 0.0041130
2: 0.0084066, 0.0193341, 0.0092692, 0.0192927, -0.0097881, 0.0089681
3: -0.0067131, -0.0017723, -0.0066708, -0.0022665, -0.0044466, 0.0048985
4: 0.0038907, 0.0055345, 0.0039042, 0.0053746, -0.0013503, 0.0014915
5: -0.0038575, -0.0002268, -0.0038431, -0.0004534, -0.0034041, 0.0036163
6: -0.0067417, -0.0049735, -0.0067294, -0.0051322, -0.0016095, 0.0017559
7: -0.0040409, -0.0005022, -0.0036275, -0.0005668, -0.0034741, 0.0031253
8: -0.0072705, -0.0000416, -0.0072464, -0.0005987, -0.0066718, 0.0072048
9: 1.0003084, 1.0023073, 1.0003716, 1.0016876, -0.0013793, 0.0019357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015226, upper bound: 0.0014674
time: 2.81 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015980, upper bound: 0.0016091
time: 1.67 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0050113, 0.0005875, -0.0057616, 0.0056504
1: 0.0032942, 0.0074643, 0.0033506, 0.0073251, -0.0040309, 0.0041137
2: 0.0084066, 0.0193341, 0.0087046, 0.0192520, -0.0096547, 0.0094388
3: -0.0067131, -0.0017723, -0.0066545, -0.0019347, -0.0047783, 0.0048821
4: 0.0038907, 0.0055345, 0.0039107, 0.0054817, -0.0013843, 0.0014185
5: -0.0038575, -0.0002268, -0.0038354, -0.0003103, -0.0035473, 0.0036086
6: -0.0067417, -0.0049735, -0.0067229, -0.0050264, -0.0017153, 0.0017494
7: -0.0040409, -0.0005022, -0.0039236, -0.0005583, -0.0034826, 0.0034214
8: -0.0072705, -0.0000416, -0.0072211, -0.0002366, -0.0070339, 0.0071795
9: 1.0003084, 1.0023073, 1.0003281, 1.0020975, -0.0017891, 0.0019792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015226, upper bound: 0.0014674
time: 1.97 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015980, upper bound: 0.0016091
time: 1.39 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0048370, 0.0006481, -0.0058222, 0.0054761
1: 0.0032942, 0.0074643, 0.0033019, 0.0071539, -0.0038598, 0.0041625
2: 0.0084066, 0.0193341, 0.0090055, 0.0193522, -0.0097747, 0.0091548
3: -0.0067131, -0.0017723, -0.0067221, -0.0021260, -0.0045871, 0.0049498
4: 0.0038907, 0.0055345, 0.0038875, 0.0054204, -0.0013617, 0.0014731
5: -0.0038575, -0.0002268, -0.0038581, -0.0003803, -0.0034772, 0.0036313
6: -0.0067417, -0.0049735, -0.0067448, -0.0050854, -0.0016563, 0.0017714
7: -0.0040409, -0.0005022, -0.0037320, -0.0005139, -0.0035270, 0.0032298
8: -0.0072705, -0.0000416, -0.0072838, -0.0004284, -0.0068421, 0.0072423
9: 1.0003084, 1.0023073, 1.0003542, 1.0018717, -0.0015633, 0.0019531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013175, upper bound: 0.0014742
time: 2.16 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016152
time: 1.30 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0051742, 0.0006392, -0.0058133, 0.0058133
1: 0.0032942, 0.0074643, 0.0032942, 0.0074643, -0.0041702, 0.0041702
2: 0.0084066, 0.0193341, 0.0084066, 0.0193341, -0.0096650, 0.0096650
3: -0.0067131, -0.0017723, -0.0067131, -0.0017723, -0.0049407, 0.0049407
4: 0.0038907, 0.0055345, 0.0038907, 0.0055345, -0.0014058, 0.0014058
5: -0.0038575, -0.0002268, -0.0038575, -0.0002268, -0.0036307, 0.0036307
6: -0.0067417, -0.0049735, -0.0067417, -0.0049735, -0.0017682, 0.0017682
7: -0.0040409, -0.0005022, -0.0040409, -0.0005022, -0.0035387, 0.0035387
8: -0.0072705, -0.0000416, -0.0072705, -0.0000416, -0.0072289, 0.0072289
9: 1.0003084, 1.0023073, 1.0003084, 1.0023073, -0.0019989, 0.0019989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015375, upper bound: 0.0014742
time: 2.06 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016152
time: 1.45 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0048278, 0.0005301, -0.0046920, 0.0006092, -0.0054370, 0.0052221
1: 0.0033941, 0.0071543, 0.0033514, 0.0070309, -0.0036368, 0.0038029
2: 0.0090253, 0.0191401, 0.0092692, 0.0192927, -0.0091523, 0.0087650
3: -0.0066109, -0.0021276, -0.0066708, -0.0022665, -0.0043444, 0.0045432
4: 0.0039253, 0.0054196, 0.0039042, 0.0053746, -0.0013389, 0.0013867
5: -0.0037876, -0.0003922, -0.0038431, -0.0004534, -0.0033343, 0.0034509
6: -0.0067061, -0.0050873, -0.0067294, -0.0051322, -0.0015739, 0.0016421
7: -0.0037476, -0.0005895, -0.0036275, -0.0005668, -0.0031809, 0.0030380
8: -0.0071428, -0.0004418, -0.0072464, -0.0005987, -0.0065442, 0.0068047
9: 1.0003535, 1.0018612, 1.0003716, 1.0016876, -0.0013342, 0.0014896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016009, upper bound: 0.0015342
time: 1.79 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016545, upper bound: 0.0016567
time: 1.21 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048278, 0.0005301, -0.0048370, 0.0006481, -0.0054759, 0.0053671
1: 0.0033941, 0.0071543, 0.0033019, 0.0071539, -0.0037598, 0.0038524
2: 0.0090253, 0.0191401, 0.0090055, 0.0193522, -0.0092348, 0.0090483
3: -0.0066109, -0.0021276, -0.0067221, -0.0021260, -0.0044849, 0.0045945
4: 0.0039253, 0.0054196, 0.0038875, 0.0054204, -0.0013833, 0.0013993
5: -0.0037876, -0.0003922, -0.0038581, -0.0003803, -0.0034073, 0.0034659
6: -0.0067061, -0.0050873, -0.0067448, -0.0050854, -0.0016207, 0.0016576
7: -0.0037476, -0.0005895, -0.0037320, -0.0005139, -0.0032338, 0.0031425
8: -0.0071428, -0.0004418, -0.0072838, -0.0004284, -0.0067145, 0.0068421
9: 1.0003535, 1.0018612, 1.0003542, 1.0018717, -0.0015182, 0.0015070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016009, upper bound: 0.0015390
time: 1.51 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016545, upper bound: 0.0016599
time: 1.59 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0051470, 0.0005027, -0.0046920, 0.0006092, -0.0057562, 0.0051947
1: 0.0033933, 0.0074547, 0.0033514, 0.0070309, -0.0036376, 0.0041033
2: 0.0084569, 0.0190914, 0.0092692, 0.0192927, -0.0097434, 0.0087399
3: -0.0065925, -0.0017877, -0.0066708, -0.0022665, -0.0043260, 0.0048831
4: 0.0039324, 0.0055289, 0.0039042, 0.0053746, -0.0013354, 0.0014976
5: -0.0037792, -0.0002434, -0.0038431, -0.0004534, -0.0033258, 0.0035997
6: -0.0066975, -0.0049808, -0.0067294, -0.0051322, -0.0015654, 0.0017486
7: -0.0040536, -0.0005844, -0.0036275, -0.0005668, -0.0034869, 0.0030431
8: -0.0071131, -0.0000742, -0.0072464, -0.0005987, -0.0065144, 0.0071722
9: 1.0003088, 1.0022757, 1.0003716, 1.0016876, -0.0013789, 0.0019041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015459, upper bound: 0.0014643
time: 1.41 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016390, upper bound: 0.0015989
time: 1.46 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0051470, 0.0005027, -0.0048370, 0.0006481, -0.0057951, 0.0053397
1: 0.0033933, 0.0074547, 0.0033019, 0.0071539, -0.0037606, 0.0041528
2: 0.0084569, 0.0190914, 0.0090055, 0.0193522, -0.0098259, 0.0090232
3: -0.0065925, -0.0017877, -0.0067221, -0.0021260, -0.0044665, 0.0049344
4: 0.0039324, 0.0055289, 0.0038875, 0.0054204, -0.0013798, 0.0015102
5: -0.0037792, -0.0002434, -0.0038581, -0.0003803, -0.0033989, 0.0036147
6: -0.0066975, -0.0049808, -0.0067448, -0.0050854, -0.0016121, 0.0017641
7: -0.0040536, -0.0005844, -0.0037320, -0.0005139, -0.0035397, 0.0031476
8: -0.0071131, -0.0000742, -0.0072838, -0.0004284, -0.0066847, 0.0072096
9: 1.0003088, 1.0022757, 1.0003542, 1.0018717, -0.0015630, 0.0019215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015459, upper bound: 0.0014696
time: 1.48 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016390, upper bound: 0.0016023
time: 1.99 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0048278, 0.0005301, -0.0050113, 0.0005875, -0.0054153, 0.0055414
1: 0.0033941, 0.0071543, 0.0033506, 0.0073251, -0.0039310, 0.0038036
2: 0.0090253, 0.0191401, 0.0087046, 0.0192520, -0.0091284, 0.0093487
3: -0.0066109, -0.0021276, -0.0066545, -0.0019347, -0.0046762, 0.0045269
4: 0.0039253, 0.0054196, 0.0039107, 0.0054817, -0.0014557, 0.0013928
5: -0.0037876, -0.0003922, -0.0038354, -0.0003103, -0.0034774, 0.0034431
6: -0.0067061, -0.0050873, -0.0067229, -0.0050264, -0.0016797, 0.0016357
7: -0.0037476, -0.0005895, -0.0039236, -0.0005583, -0.0031893, 0.0033341
8: -0.0071428, -0.0004418, -0.0072211, -0.0002366, -0.0069063, 0.0067793
9: 1.0003535, 1.0018612, 1.0003281, 1.0020975, -0.0017440, 0.0015332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014613, upper bound: 0.0015201
time: 1.41 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015951, upper bound: 0.0015957
time: 1.23 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0051470, 0.0005027, -0.0050113, 0.0005875, -0.0057345, 0.0055140
1: 0.0033933, 0.0074547, 0.0033506, 0.0073251, -0.0039318, 0.0041040
2: 0.0084569, 0.0190914, 0.0087046, 0.0192520, -0.0096187, 0.0092197
3: -0.0065925, -0.0017877, -0.0066545, -0.0019347, -0.0046578, 0.0048668
4: 0.0039324, 0.0055289, 0.0039107, 0.0054817, -0.0013864, 0.0014333
5: -0.0037792, -0.0002434, -0.0038354, -0.0003103, -0.0034689, 0.0035920
6: -0.0066975, -0.0049808, -0.0067229, -0.0050264, -0.0016711, 0.0017421
7: -0.0040536, -0.0005844, -0.0039236, -0.0005583, -0.0034953, 0.0033392
8: -0.0071131, -0.0000742, -0.0072211, -0.0002366, -0.0068765, 0.0071468
9: 1.0003088, 1.0022757, 1.0003281, 1.0020975, -0.0017887, 0.0019476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015151, upper bound: 0.0014587
time: 1.63 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015951, upper bound: 0.0015957
time: 1.46 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0048278, 0.0005301, -0.0051742, 0.0006392, -0.0054669, 0.0057042
1: 0.0033941, 0.0071543, 0.0032942, 0.0074643, -0.0040702, 0.0038601
2: 0.0090253, 0.0191401, 0.0084066, 0.0193341, -0.0092474, 0.0096801
3: -0.0066109, -0.0021276, -0.0067131, -0.0017723, -0.0048386, 0.0045855
4: 0.0039253, 0.0054196, 0.0038907, 0.0055345, -0.0015081, 0.0014147
5: -0.0037876, -0.0003922, -0.0038575, -0.0002268, -0.0035608, 0.0034653
6: -0.0067061, -0.0050873, -0.0067417, -0.0049735, -0.0017326, 0.0016544
7: -0.0037476, -0.0005895, -0.0040409, -0.0005022, -0.0032454, 0.0034514
8: -0.0071428, -0.0004418, -0.0072705, -0.0000416, -0.0071013, 0.0068287
9: 1.0003535, 1.0018612, 1.0003084, 1.0023073, -0.0019538, 0.0015528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014664, upper bound: 0.0015249
time: 1.52 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016082, upper bound: 0.0015983
time: 1.27 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0051470, 0.0005027, -0.0051742, 0.0006392, -0.0057862, 0.0056769
1: 0.0033933, 0.0074547, 0.0032942, 0.0074643, -0.0040710, 0.0041605
2: 0.0084569, 0.0190914, 0.0084066, 0.0193341, -0.0097302, 0.0095471
3: -0.0065925, -0.0017877, -0.0067131, -0.0017723, -0.0048202, 0.0049253
4: 0.0039324, 0.0055289, 0.0038907, 0.0055345, -0.0014339, 0.0014465
5: -0.0037792, -0.0002434, -0.0038575, -0.0002268, -0.0035524, 0.0036141
6: -0.0066975, -0.0049808, -0.0067417, -0.0049735, -0.0017240, 0.0017609
7: -0.0040536, -0.0005844, -0.0040409, -0.0005022, -0.0035514, 0.0034565
8: -0.0071131, -0.0000742, -0.0072705, -0.0000416, -0.0070715, 0.0071962
9: 1.0003088, 1.0022757, 1.0003084, 1.0023073, -0.0019985, 0.0019673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015222, upper bound: 0.0014625
time: 1.94 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016082, upper bound: 0.0015983
time: 1.24 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0046920, 0.0006092, -0.0055667, 0.0052470
1: 0.0033580, 0.0072629, 0.0033514, 0.0070309, -0.0036728, 0.0039115
2: 0.0087894, 0.0191808, 0.0092692, 0.0192927, -0.0093942, 0.0088093
3: -0.0066432, -0.0020018, -0.0066708, -0.0022665, -0.0043767, 0.0046690
4: 0.0039155, 0.0054603, 0.0039042, 0.0053746, -0.0013445, 0.0014232
5: -0.0037978, -0.0003250, -0.0038431, -0.0004534, -0.0033444, 0.0035181
6: -0.0067153, -0.0050458, -0.0067294, -0.0051322, -0.0015831, 0.0016835
7: -0.0038447, -0.0005489, -0.0036275, -0.0005668, -0.0032779, 0.0030786
8: -0.0071707, -0.0002886, -0.0072464, -0.0005987, -0.0065721, 0.0069578
9: 1.0003382, 1.0020260, 1.0003716, 1.0016876, -0.0013494, 0.0016544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015507, upper bound: 0.0014713
time: 1.79 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016401, upper bound: 0.0016119
time: 1.52 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0052958, 0.0005380, -0.0046920, 0.0006092, -0.0059049, 0.0052301
1: 0.0033464, 0.0075762, 0.0033514, 0.0070309, -0.0036845, 0.0042248
2: 0.0081890, 0.0191475, 0.0092692, 0.0192927, -0.0100215, 0.0088133
3: -0.0066350, -0.0016461, -0.0066708, -0.0022665, -0.0043685, 0.0050248
4: 0.0039192, 0.0055750, 0.0039042, 0.0053746, -0.0013499, 0.0015418
5: -0.0037958, -0.0001657, -0.0038431, -0.0004534, -0.0033424, 0.0036775
6: -0.0067104, -0.0049336, -0.0067294, -0.0051322, -0.0015782, 0.0017957
7: -0.0041612, -0.0005394, -0.0036275, -0.0005668, -0.0035945, 0.0030881
8: -0.0071495, 0.0000998, -0.0072464, -0.0005987, -0.0065509, 0.0073462
9: 1.0002913, 1.0024639, 1.0003716, 1.0016876, -0.0013963, 0.0020924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015507, upper bound: 0.0014713
time: 1.37 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016401, upper bound: 0.0016119
time: 1.53 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0048370, 0.0006481, -0.0056056, 0.0053919
1: 0.0033580, 0.0072629, 0.0033019, 0.0071539, -0.0037959, 0.0039610
2: 0.0087894, 0.0191808, 0.0090055, 0.0193522, -0.0094007, 0.0090173
3: -0.0066432, -0.0020018, -0.0067221, -0.0021260, -0.0045172, 0.0047203
4: 0.0039155, 0.0054603, 0.0038875, 0.0054204, -0.0013700, 0.0014170
5: -0.0037978, -0.0003250, -0.0038581, -0.0003803, -0.0034175, 0.0035331
6: -0.0067153, -0.0050458, -0.0067448, -0.0050854, -0.0016299, 0.0016990
7: -0.0038447, -0.0005489, -0.0037320, -0.0005139, -0.0033308, 0.0031830
8: -0.0071707, -0.0002886, -0.0072838, -0.0004284, -0.0067423, 0.0069952
9: 1.0003382, 1.0020260, 1.0003542, 1.0018717, -0.0015335, 0.0016718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015757, upper bound: 0.0014784
time: 1.96 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016754, upper bound: 0.0016184
time: 2.04 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0052958, 0.0005380, -0.0048370, 0.0006481, -0.0059438, 0.0053750
1: 0.0033464, 0.0075762, 0.0033019, 0.0071539, -0.0038076, 0.0042743
2: 0.0081890, 0.0191475, 0.0090055, 0.0193522, -0.0100228, 0.0090178
3: -0.0066350, -0.0016461, -0.0067221, -0.0021260, -0.0045090, 0.0050761
4: 0.0039192, 0.0055750, 0.0038875, 0.0054204, -0.0013680, 0.0015302
5: -0.0037958, -0.0001657, -0.0038581, -0.0003803, -0.0034155, 0.0036925
6: -0.0067104, -0.0049336, -0.0067448, -0.0050854, -0.0016250, 0.0018112
7: -0.0041612, -0.0005394, -0.0037320, -0.0005139, -0.0036474, 0.0031926
8: -0.0071495, 0.0000998, -0.0072838, -0.0004284, -0.0067212, 0.0073837
9: 1.0002913, 1.0024639, 1.0003542, 1.0018717, -0.0015804, 0.0021098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015757, upper bound: 0.0014784
time: 1.64 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016754, upper bound: 0.0016184
time: 1.86 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0050113, 0.0005875, -0.0055450, 0.0055662
1: 0.0033580, 0.0072629, 0.0033506, 0.0073251, -0.0039670, 0.0039122
2: 0.0087894, 0.0191808, 0.0087046, 0.0192520, -0.0093703, 0.0093930
3: -0.0066432, -0.0020018, -0.0066545, -0.0019347, -0.0047084, 0.0046526
4: 0.0039155, 0.0054603, 0.0039107, 0.0054817, -0.0014613, 0.0014293
5: -0.0037978, -0.0003250, -0.0038354, -0.0003103, -0.0034876, 0.0035103
6: -0.0067153, -0.0050458, -0.0067229, -0.0050264, -0.0016889, 0.0016771
7: -0.0038447, -0.0005489, -0.0039236, -0.0005583, -0.0032864, 0.0033747
8: -0.0071707, -0.0002886, -0.0072211, -0.0002366, -0.0069342, 0.0069325
9: 1.0003382, 1.0020260, 1.0003281, 1.0020975, -0.0017593, 0.0016979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014649, upper bound: 0.0015279
time: 1.29 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015970, upper bound: 0.0016091
time: 1.69 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0052958, 0.0005380, -0.0050113, 0.0005875, -0.0058832, 0.0055493
1: 0.0033464, 0.0075762, 0.0033506, 0.0073251, -0.0039787, 0.0042255
2: 0.0081890, 0.0191475, 0.0087046, 0.0192520, -0.0098903, 0.0092844
3: -0.0066350, -0.0016461, -0.0066545, -0.0019347, -0.0047003, 0.0050084
4: 0.0039192, 0.0055750, 0.0039107, 0.0054817, -0.0013924, 0.0014723
5: -0.0037958, -0.0001657, -0.0038354, -0.0003103, -0.0034855, 0.0036697
6: -0.0067104, -0.0049336, -0.0067229, -0.0050264, -0.0016840, 0.0017893
7: -0.0041612, -0.0005394, -0.0039236, -0.0005583, -0.0036029, 0.0033842
8: -0.0071495, 0.0000998, -0.0072211, -0.0002366, -0.0069130, 0.0073209
9: 1.0002913, 1.0024639, 1.0003281, 1.0020975, -0.0018061, 0.0021359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015208, upper bound: 0.0014657
time: 2.25 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015970, upper bound: 0.0016091
time: 1.39 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0051742, 0.0006392, -0.0055967, 0.0057291
1: 0.0033580, 0.0072629, 0.0032942, 0.0074643, -0.0041063, 0.0039687
2: 0.0087894, 0.0191808, 0.0084066, 0.0193341, -0.0094045, 0.0096410
3: -0.0066432, -0.0020018, -0.0067131, -0.0017723, -0.0048708, 0.0047112
4: 0.0039155, 0.0054603, 0.0038907, 0.0055345, -0.0014889, 0.0014246
5: -0.0037978, -0.0003250, -0.0038575, -0.0002268, -0.0035710, 0.0035325
6: -0.0067153, -0.0050458, -0.0067417, -0.0049735, -0.0017418, 0.0016958
7: -0.0038447, -0.0005489, -0.0040409, -0.0005022, -0.0033425, 0.0034920
8: -0.0071707, -0.0002886, -0.0072705, -0.0000416, -0.0071292, 0.0069819
9: 1.0003382, 1.0020260, 1.0003084, 1.0023073, -0.0019691, 0.0017176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012590, upper bound: 0.0015416
time: 2.48 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016139, upper bound: 0.0016152
time: 1.57 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0052958, 0.0005380, -0.0051742, 0.0006392, -0.0059349, 0.0057122
1: 0.0033464, 0.0075762, 0.0032942, 0.0074643, -0.0041179, 0.0042820
2: 0.0081890, 0.0191475, 0.0084066, 0.0193341, -0.0099179, 0.0095284
3: -0.0066350, -0.0016461, -0.0067131, -0.0017723, -0.0048627, 0.0050670
4: 0.0039192, 0.0055750, 0.0038907, 0.0055345, -0.0014204, 0.0014664
5: -0.0037958, -0.0001657, -0.0038575, -0.0002268, -0.0035690, 0.0036918
6: -0.0067104, -0.0049336, -0.0067417, -0.0049735, -0.0017369, 0.0018081
7: -0.0041612, -0.0005394, -0.0040409, -0.0005022, -0.0036591, 0.0035015
8: -0.0071495, 0.0000998, -0.0072705, -0.0000416, -0.0071080, 0.0073703
9: 1.0002913, 1.0024639, 1.0003084, 1.0023073, -0.0020159, 0.0021555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015359, upper bound: 0.0014727
time: 1.95 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016139, upper bound: 0.0016152
time: 1.47 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.70 seconds
IS_B1_A1_A2_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016015, upper bound: 0.0015349
IS_B1_A1_A2_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016552, upper bound: 0.0016569
IS_B1_A1_A2_B2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016119
IS_B1_A1_A2_B2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016552, upper bound: 0.0016601
IS_B1_A1_A2_B2_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0014682, upper bound: 0.0015568
IS_B1_A1_A2_B2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015987, upper bound: 0.0016414
IS_B1_A1_A2_B2_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0014682, upper bound: 0.0015596
IS_B1_A1_A2_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015987, upper bound: 0.0016433
IS_B1_A1_A2_B2_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015187, upper bound: 0.0014621
IS_B1_A1_A2_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015958, upper bound: 0.0015962
IS_B1_A1_A2_B2_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015187, upper bound: 0.0014621
IS_B1_A1_A2_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015958, upper bound: 0.0015962
IS_B1_A1_A2_B2_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0014659
IS_B1_A1_A2_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015987
IS_B1_A1_A2_B2_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0014674, upper bound: 0.0015253
IS_B1_A1_A2_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015987
IS_B1_A1_A2_B2_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
IS_B1_A1_A2_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016013, upper bound: 0.0016726
IS_B1_A1_A2_B2_A2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0015775
IS_B1_A1_A2_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016013, upper bound: 0.0016726
IS_B1_A1_A2_B2_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0015418
IS_B1_A1_A2_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016829
IS_B1_A1_A2_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0014823, upper bound: 0.0015904
IS_B1_A1_A2_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016829
IS_B1_A1_A2_B2_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015226, upper bound: 0.0014674
IS_B1_A1_A2_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015980, upper bound: 0.0016091
IS_B1_A1_A2_B2_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015226, upper bound: 0.0014674
IS_B1_A1_A2_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015980, upper bound: 0.0016091
IS_B1_A1_A2_B2_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0013175, upper bound: 0.0014742
IS_B1_A1_A2_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016152
IS_B1_A1_A2_B2_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015375, upper bound: 0.0014742
IS_B1_A1_A2_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016152
IS_B1_A2_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016009, upper bound: 0.0015342
IS_B1_A2_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016545, upper bound: 0.0016567
IS_B1_A2_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016009, upper bound: 0.0015390
IS_B1_A2_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016545, upper bound: 0.0016599
IS_B1_A2_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015459, upper bound: 0.0014643
IS_B1_A2_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016390, upper bound: 0.0015989
IS_B1_A2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015459, upper bound: 0.0014696
IS_B1_A2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016390, upper bound: 0.0016023
IS_B1_A2_A2_B2_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0014613, upper bound: 0.0015201
IS_B1_A2_A2_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015951, upper bound: 0.0015957
IS_B1_A2_A2_B2_A1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015151, upper bound: 0.0014587
IS_B1_A2_A2_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015951, upper bound: 0.0015957
IS_B1_A2_A2_B2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0014664, upper bound: 0.0015249
IS_B1_A2_A2_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016082, upper bound: 0.0015983
IS_B1_A2_A2_B2_A1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015222, upper bound: 0.0014625
IS_B1_A2_A2_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016082, upper bound: 0.0015983
IS_B1_A2_A2_B2_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015507, upper bound: 0.0014713
IS_B1_A2_A2_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016401, upper bound: 0.0016119
IS_B1_A2_A2_B2_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015507, upper bound: 0.0014713
IS_B1_A2_A2_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016401, upper bound: 0.0016119
IS_B1_A2_A2_B2_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015757, upper bound: 0.0014784
IS_B1_A2_A2_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016754, upper bound: 0.0016184
IS_B1_A2_A2_B2_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015757, upper bound: 0.0014784
IS_B1_A2_A2_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016754, upper bound: 0.0016184
IS_B1_A2_A2_B2_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0014649, upper bound: 0.0015279
IS_B1_A2_A2_B2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015970, upper bound: 0.0016091
IS_B1_A2_A2_B2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015208, upper bound: 0.0014657
IS_B1_A2_A2_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015970, upper bound: 0.0016091
IS_B1_A2_A2_B2_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0012590, upper bound: 0.0015416
IS_B1_A2_A2_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016139, upper bound: 0.0016152
IS_B1_A2_A2_B2_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0015359, upper bound: 0.0014727
IS_B1_A2_A2_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.70
Output dim: 9, lower bound: -0.0016139, upper bound: 0.0016152

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0044601, 0.0006993, -0.0046574, 0.0006087, -0.0050688, 0.0053567
1: 0.0032986, 0.0068191, 0.0033532, 0.0069992, -0.0037005, 0.0034659
2: 0.0096808, 0.0194604, 0.0093304, 0.0192917, -0.0084624, 0.0089647
3: -0.0067452, -0.0025091, -0.0066704, -0.0023028, -0.0044424, 0.0041613
4: 0.0038797, 0.0052965, 0.0039043, 0.0053630, -0.0013322, 0.0012455
5: -0.0038883, -0.0005638, -0.0038424, -0.0004697, -0.0034186, 0.0032786
6: -0.0067565, -0.0052091, -0.0067292, -0.0051435, -0.0016130, 0.0015202
7: -0.0034129, -0.0005372, -0.0035950, -0.0005694, -0.0028435, 0.0030578
8: -0.0073556, -0.0008654, -0.0072456, -0.0006384, -0.0067172, 0.0063802
9: 1.0004028, 1.0013897, 1.0003765, 1.0016433, -0.0012405, 0.0010133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015206, upper bound: 0.0015207
time: 1.09 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015206, upper bound: 0.0015349
time: 1.13 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0046920, 0.0006092, -0.0052397, 0.0053000
1: 0.0033547, 0.0069801, 0.0033514, 0.0070309, -0.0036761, 0.0036288
2: 0.0093798, 0.0192902, 0.0092692, 0.0192927, -0.0087367, 0.0088695
3: -0.0066698, -0.0023249, -0.0066708, -0.0022665, -0.0044033, 0.0043460
4: 0.0039045, 0.0053556, 0.0039042, 0.0053746, -0.0013220, 0.0012903
5: -0.0038415, -0.0004855, -0.0038431, -0.0004534, -0.0033881, 0.0033577
6: -0.0067290, -0.0051519, -0.0067294, -0.0051322, -0.0015969, 0.0015775
7: -0.0035825, -0.0005714, -0.0036275, -0.0005668, -0.0030158, 0.0030561
8: -0.0072445, -0.0006709, -0.0072464, -0.0005987, -0.0066459, 0.0065755
9: 1.0003786, 1.0016094, 1.0003716, 1.0016876, -0.0013090, 0.0012379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016040
time: 1.54 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016569
time: 1.32 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0046574, 0.0006087, -0.0045990, 0.0007331, -0.0053905, 0.0052076
1: 0.0033532, 0.0069992, 0.0032527, 0.0069358, -0.0035826, 0.0037465
2: 0.0093304, 0.0192917, 0.0094285, 0.0195143, -0.0090259, 0.0087259
3: -0.0066704, -0.0023028, -0.0067927, -0.0023730, -0.0042974, 0.0044899
4: 0.0039043, 0.0053630, 0.0038640, 0.0053411, -0.0012892, 0.0013389
5: -0.0038424, -0.0004697, -0.0039016, -0.0004934, -0.0033490, 0.0034319
6: -0.0067292, -0.0051435, -0.0067698, -0.0051638, -0.0015654, 0.0016263
7: -0.0035950, -0.0005694, -0.0035193, -0.0004877, -0.0031073, 0.0029499
8: -0.0072456, -0.0006384, -0.0073887, -0.0007017, -0.0065439, 0.0067503
9: 1.0003765, 1.0016433, 1.0003860, 1.0015668, -0.0011903, 0.0012573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015461, upper bound: 0.0015279
time: 1.19 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015461, upper bound: 0.0015279
time: 1.21 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0047777, 0.0006468, -0.0053388, 0.0053869
1: 0.0033514, 0.0070309, 0.0033051, 0.0071064, -0.0037551, 0.0037257
2: 0.0092692, 0.0192927, 0.0091127, 0.0193497, -0.0089523, 0.0090192
3: -0.0066708, -0.0022665, -0.0067212, -0.0021812, -0.0044896, 0.0044548
4: 0.0039042, 0.0053746, 0.0038877, 0.0054023, -0.0013345, 0.0013345
5: -0.0038431, -0.0004534, -0.0038564, -0.0004109, -0.0034322, 0.0034030
6: -0.0067294, -0.0051322, -0.0067445, -0.0051042, -0.0016251, 0.0016124
7: -0.0036275, -0.0005668, -0.0036876, -0.0005181, -0.0031094, 0.0031209
8: -0.0072464, -0.0005987, -0.0072819, -0.0004982, -0.0067482, 0.0066833
9: 1.0003716, 1.0016876, 1.0003608, 1.0017968, -0.0014253, 0.0013268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016259, upper bound: 0.0015397
time: 1.59 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016259, upper bound: 0.0015397
time: 1.40 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0049481, 0.0005859, -0.0052779, 0.0055573
1: 0.0033514, 0.0070309, 0.0033546, 0.0072754, -0.0039240, 0.0036763
2: 0.0092692, 0.0192927, 0.0088181, 0.0192488, -0.0088452, 0.0093190
3: -0.0066708, -0.0022665, -0.0066533, -0.0019928, -0.0046780, 0.0043868
4: 0.0039042, 0.0053746, 0.0039111, 0.0054625, -0.0014087, 0.0013280
5: -0.0038431, -0.0004534, -0.0038333, -0.0003428, -0.0035003, 0.0033799
6: -0.0067294, -0.0051322, -0.0067225, -0.0050463, -0.0016830, 0.0015904
7: -0.0036275, -0.0005668, -0.0038787, -0.0005631, -0.0030644, 0.0033119
8: -0.0072464, -0.0005987, -0.0072188, -0.0003103, -0.0069361, 0.0066201
9: 1.0003716, 1.0016876, 1.0003350, 1.0020182, -0.0016466, 0.0013527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0015112
time: 1.56 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0016414
time: 3.13 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0051156, 0.0006376, -0.0053296, 0.0057248
1: 0.0033514, 0.0070309, 0.0032984, 0.0074150, -0.0040637, 0.0037324
2: 0.0092692, 0.0192927, 0.0085128, 0.0193310, -0.0089642, 0.0096503
3: -0.0066708, -0.0022665, -0.0067119, -0.0018286, -0.0048422, 0.0044454
4: 0.0039042, 0.0053746, 0.0038911, 0.0055163, -0.0014581, 0.0013499
5: -0.0038431, -0.0004534, -0.0038553, -0.0002577, -0.0035855, 0.0034019
6: -0.0067294, -0.0051322, -0.0067413, -0.0049922, -0.0017372, 0.0016091
7: -0.0036275, -0.0005668, -0.0039943, -0.0005063, -0.0031212, 0.0034276
8: -0.0072464, -0.0005987, -0.0072682, -0.0001106, -0.0071358, 0.0066695
9: 1.0003716, 1.0016876, 1.0003155, 1.0022328, -0.0018612, 0.0013721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015401, upper bound: 0.0015133
time: 1.27 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015401, upper bound: 0.0016433
time: 1.66 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0046920, 0.0006092, -0.0055573, 0.0052779
1: 0.0033546, 0.0072754, 0.0033514, 0.0070309, -0.0036763, 0.0039240
2: 0.0088181, 0.0192488, 0.0092692, 0.0192927, -0.0093190, 0.0088452
3: -0.0066533, -0.0019928, -0.0066708, -0.0022665, -0.0043868, 0.0046780
4: 0.0039111, 0.0054625, 0.0039042, 0.0053746, -0.0013280, 0.0014087
5: -0.0038333, -0.0003428, -0.0038431, -0.0004534, -0.0033799, 0.0035003
6: -0.0067225, -0.0050463, -0.0067294, -0.0051322, -0.0015904, 0.0016830
7: -0.0038787, -0.0005631, -0.0036275, -0.0005668, -0.0033119, 0.0030644
8: -0.0072188, -0.0003103, -0.0072464, -0.0005987, -0.0066201, 0.0069361
9: 1.0003350, 1.0020182, 1.0003716, 1.0016876, -0.0013527, 0.0016466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015109, upper bound: 0.0015349
time: 1.40 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015109, upper bound: 0.0015992
time: 1.24 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0050113, 0.0005875, -0.0055356, 0.0055972
1: 0.0033546, 0.0072754, 0.0033506, 0.0073251, -0.0039705, 0.0039247
2: 0.0088181, 0.0192488, 0.0087046, 0.0192520, -0.0091882, 0.0093233
3: -0.0066533, -0.0019928, -0.0066545, -0.0019347, -0.0047186, 0.0046616
4: 0.0039111, 0.0054625, 0.0039107, 0.0054817, -0.0013707, 0.0013375
5: -0.0038333, -0.0003428, -0.0038354, -0.0003103, -0.0035231, 0.0034925
6: -0.0067225, -0.0050463, -0.0067229, -0.0050264, -0.0016961, 0.0016766
7: -0.0038787, -0.0005631, -0.0039236, -0.0005583, -0.0033204, 0.0033605
8: -0.0072188, -0.0003103, -0.0072211, -0.0002366, -0.0069822, 0.0069107
9: 1.0003350, 1.0020182, 1.0003281, 1.0020975, -0.0017625, 0.0016901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014622, upper bound: 0.0015208
time: 1.23 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014622, upper bound: 0.0015962
time: 1.26 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0048370, 0.0006481, -0.0055962, 0.0054229
1: 0.0033546, 0.0072754, 0.0033019, 0.0071539, -0.0037994, 0.0039735
2: 0.0088181, 0.0192488, 0.0090055, 0.0193522, -0.0093919, 0.0091285
3: -0.0066533, -0.0019928, -0.0067221, -0.0021260, -0.0045273, 0.0047293
4: 0.0039111, 0.0054625, 0.0038875, 0.0054204, -0.0013724, 0.0014166
5: -0.0038333, -0.0003428, -0.0038581, -0.0003803, -0.0034530, 0.0035153
6: -0.0067225, -0.0050463, -0.0067448, -0.0050854, -0.0016371, 0.0016985
7: -0.0038787, -0.0005631, -0.0037320, -0.0005139, -0.0033648, 0.0031689
8: -0.0072188, -0.0003103, -0.0072838, -0.0004284, -0.0067904, 0.0069735
9: 1.0003350, 1.0020182, 1.0003542, 1.0018717, -0.0015367, 0.0016640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0015435
time: 1.07 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0016027
time: 1.56 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0051156, 0.0006376, -0.0056489, 0.0057031
1: 0.0033506, 0.0073251, 0.0032984, 0.0074150, -0.0040644, 0.0040267
2: 0.0087046, 0.0192520, 0.0085128, 0.0193310, -0.0094349, 0.0095148
3: -0.0066545, -0.0019347, -0.0067119, -0.0018286, -0.0048258, 0.0047772
4: 0.0039107, 0.0054817, 0.0038911, 0.0055163, -0.0013836, 0.0013840
5: -0.0038354, -0.0003103, -0.0038553, -0.0002577, -0.0035777, 0.0035450
6: -0.0067229, -0.0050264, -0.0067413, -0.0049922, -0.0017307, 0.0017149
7: -0.0039236, -0.0005583, -0.0039943, -0.0005063, -0.0034173, 0.0034360
8: -0.0072211, -0.0002366, -0.0072682, -0.0001106, -0.0071104, 0.0070316
9: 1.0003281, 1.0020975, 1.0003155, 1.0022328, -0.0019047, 0.0017819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0014659
time: 1.46 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0015987
time: 2.01 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0046920, 0.0006092, -0.0053869, 0.0053388
1: 0.0033051, 0.0071064, 0.0033514, 0.0070309, -0.0037257, 0.0037551
2: 0.0091127, 0.0193497, 0.0092692, 0.0192927, -0.0090192, 0.0089523
3: -0.0067212, -0.0021812, -0.0066708, -0.0022665, -0.0044548, 0.0044896
4: 0.0038877, 0.0054023, 0.0039042, 0.0053746, -0.0013345, 0.0013345
5: -0.0038564, -0.0004109, -0.0038431, -0.0004534, -0.0034030, 0.0034322
6: -0.0067445, -0.0051042, -0.0067294, -0.0051322, -0.0016124, 0.0016251
7: -0.0036876, -0.0005181, -0.0036275, -0.0005668, -0.0031209, 0.0031094
8: -0.0072819, -0.0004982, -0.0072464, -0.0005987, -0.0066833, 0.0067482
9: 1.0003608, 1.0017968, 1.0003716, 1.0016876, -0.0013268, 0.0014253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016298
time: 1.18 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016935
time: 1.57 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0049481, 0.0005859, -0.0054229, 0.0055962
1: 0.0033019, 0.0071539, 0.0033546, 0.0072754, -0.0039735, 0.0037994
2: 0.0090055, 0.0193522, 0.0088181, 0.0192488, -0.0091285, 0.0093919
3: -0.0067221, -0.0021260, -0.0066533, -0.0019928, -0.0047293, 0.0045273
4: 0.0038875, 0.0054204, 0.0039111, 0.0054625, -0.0014166, 0.0013724
5: -0.0038581, -0.0003803, -0.0038333, -0.0003428, -0.0035153, 0.0034530
6: -0.0067448, -0.0050854, -0.0067225, -0.0050463, -0.0016985, 0.0016371
7: -0.0037320, -0.0005139, -0.0038787, -0.0005631, -0.0031689, 0.0033648
8: -0.0072838, -0.0004284, -0.0072188, -0.0003103, -0.0069735, 0.0067904
9: 1.0003542, 1.0018717, 1.0003350, 1.0020182, -0.0016640, 0.0015367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
time: 1.78 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
time: 1.30 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0048370, 0.0006481, -0.0054258, 0.0054838
1: 0.0033051, 0.0071064, 0.0033019, 0.0071539, -0.0038488, 0.0038046
2: 0.0091127, 0.0193497, 0.0090055, 0.0193522, -0.0090066, 0.0091475
3: -0.0067212, -0.0021812, -0.0067221, -0.0021260, -0.0045952, 0.0045409
4: 0.0038877, 0.0054023, 0.0038875, 0.0054204, -0.0013538, 0.0013183
5: -0.0038564, -0.0004109, -0.0038581, -0.0003803, -0.0034761, 0.0034472
6: -0.0067445, -0.0051042, -0.0067448, -0.0050854, -0.0016591, 0.0016406
7: -0.0036876, -0.0005181, -0.0037320, -0.0005139, -0.0031738, 0.0032139
8: -0.0072819, -0.0004982, -0.0072838, -0.0004284, -0.0068536, 0.0067857
9: 1.0003608, 1.0017968, 1.0003542, 1.0018717, -0.0015109, 0.0014427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0016576
time: 1.40 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0017067
time: 1.76 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0048014, 0.0006476, -0.0049368, 0.0007195, -0.0055209, 0.0055844
1: 0.0033038, 0.0071209, 0.0032470, 0.0072450, -0.0039412, 0.0038739
2: 0.0090692, 0.0193512, 0.0088282, 0.0194851, -0.0092194, 0.0093375
3: -0.0067218, -0.0021633, -0.0067850, -0.0020222, -0.0046995, 0.0046217
4: 0.0038876, 0.0054085, 0.0038678, 0.0054538, -0.0013966, 0.0013694
5: -0.0038573, -0.0003970, -0.0038992, -0.0003398, -0.0035174, 0.0035021
6: -0.0067447, -0.0050971, -0.0067663, -0.0050529, -0.0016918, 0.0016692
7: -0.0036994, -0.0005165, -0.0038287, -0.0004730, -0.0032264, 0.0033122
8: -0.0072830, -0.0004692, -0.0073708, -0.0003137, -0.0069693, 0.0069015
9: 1.0003588, 1.0018258, 1.0003403, 1.0019988, -0.0016400, 0.0014855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014681, upper bound: 0.0015151
time: 1.24 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014681, upper bound: 0.0015151
time: 1.22 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0051156, 0.0006376, -0.0054746, 0.0057637
1: 0.0033019, 0.0071539, 0.0032984, 0.0074150, -0.0041131, 0.0038555
2: 0.0090055, 0.0193522, 0.0085128, 0.0193310, -0.0091509, 0.0096298
3: -0.0067221, -0.0021260, -0.0067119, -0.0018286, -0.0048935, 0.0045859
4: 0.0038875, 0.0054204, 0.0038911, 0.0055163, -0.0014386, 0.0013613
5: -0.0038581, -0.0003803, -0.0038553, -0.0002577, -0.0036005, 0.0034750
6: -0.0067448, -0.0050854, -0.0067413, -0.0049922, -0.0017526, 0.0016559
7: -0.0037320, -0.0005139, -0.0039943, -0.0005063, -0.0032256, 0.0034805
8: -0.0072838, -0.0004284, -0.0072682, -0.0001106, -0.0071732, 0.0068398
9: 1.0003542, 1.0018717, 1.0003155, 1.0022328, -0.0018786, 0.0015562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013112, upper bound: 0.0015418
time: 1.87 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0016829
time: 1.88 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0051156, 0.0006376, -0.0046920, 0.0006092, -0.0057248, 0.0053296
1: 0.0032984, 0.0074150, 0.0033514, 0.0070309, -0.0037324, 0.0040637
2: 0.0085128, 0.0193310, 0.0092692, 0.0192927, -0.0096503, 0.0089642
3: -0.0067119, -0.0018286, -0.0066708, -0.0022665, -0.0044454, 0.0048422
4: 0.0038911, 0.0055163, 0.0039042, 0.0053746, -0.0013499, 0.0014581
5: -0.0038553, -0.0002577, -0.0038431, -0.0004534, -0.0034019, 0.0035855
6: -0.0067413, -0.0049922, -0.0067294, -0.0051322, -0.0016091, 0.0017372
7: -0.0039943, -0.0005063, -0.0036275, -0.0005668, -0.0034276, 0.0031212
8: -0.0072682, -0.0001106, -0.0072464, -0.0005987, -0.0066695, 0.0071358
9: 1.0003155, 1.0022328, 1.0003716, 1.0016876, -0.0013721, 0.0018612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0015413
time: 1.55 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0016119
time: 2.10 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0051156, 0.0006376, -0.0050113, 0.0005875, -0.0057031, 0.0056489
1: 0.0032984, 0.0074150, 0.0033506, 0.0073251, -0.0040267, 0.0040644
2: 0.0085128, 0.0193310, 0.0087046, 0.0192520, -0.0095148, 0.0094349
3: -0.0067119, -0.0018286, -0.0066545, -0.0019347, -0.0047772, 0.0048258
4: 0.0038911, 0.0055163, 0.0039107, 0.0054817, -0.0013840, 0.0013836
5: -0.0038553, -0.0002577, -0.0038354, -0.0003103, -0.0035450, 0.0035777
6: -0.0067413, -0.0049922, -0.0067229, -0.0050264, -0.0017149, 0.0017307
7: -0.0039943, -0.0005063, -0.0039236, -0.0005583, -0.0034360, 0.0034173
8: -0.0072682, -0.0001106, -0.0072211, -0.0002366, -0.0070316, 0.0071104
9: 1.0003155, 1.0022328, 1.0003281, 1.0020975, -0.0017819, 0.0019047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0015281
time: 1.35 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0016091
time: 1.36 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0051156, 0.0006376, -0.0048370, 0.0006481, -0.0057637, 0.0054746
1: 0.0032984, 0.0074150, 0.0033019, 0.0071539, -0.0038555, 0.0041131
2: 0.0085128, 0.0193310, 0.0090055, 0.0193522, -0.0096298, 0.0091509
3: -0.0067119, -0.0018286, -0.0067221, -0.0021260, -0.0045859, 0.0048935
4: 0.0038911, 0.0055163, 0.0038875, 0.0054204, -0.0013613, 0.0014386
5: -0.0038553, -0.0002577, -0.0038581, -0.0003803, -0.0034750, 0.0036005
6: -0.0067413, -0.0049922, -0.0067448, -0.0050854, -0.0016559, 0.0017526
7: -0.0039943, -0.0005063, -0.0037320, -0.0005139, -0.0034805, 0.0032256
8: -0.0072682, -0.0001106, -0.0072838, -0.0004284, -0.0068398, 0.0071732
9: 1.0003155, 1.0022328, 1.0003542, 1.0018717, -0.0015562, 0.0018786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0015579
time: 1.23 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0016184
time: 1.91 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0051156, 0.0006376, -0.0051742, 0.0006392, -0.0057548, 0.0058117
1: 0.0032984, 0.0074150, 0.0032942, 0.0074643, -0.0041659, 0.0041208
2: 0.0085128, 0.0193310, 0.0084066, 0.0193341, -0.0095140, 0.0096612
3: -0.0067119, -0.0018286, -0.0067131, -0.0017723, -0.0049396, 0.0048844
4: 0.0038911, 0.0055163, 0.0038907, 0.0055345, -0.0014054, 0.0013687
5: -0.0038553, -0.0002577, -0.0038575, -0.0002268, -0.0036285, 0.0035998
6: -0.0067413, -0.0049922, -0.0067417, -0.0049735, -0.0017678, 0.0017495
7: -0.0039943, -0.0005063, -0.0040409, -0.0005022, -0.0034921, 0.0035345
8: -0.0072682, -0.0001106, -0.0072705, -0.0000416, -0.0072266, 0.0071598
9: 1.0003155, 1.0022328, 1.0003084, 1.0023073, -0.0019917, 0.0019244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0015417
time: 1.76 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0016152
time: 1.82 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0045823, 0.0005985, -0.0046574, 0.0006087, -0.0051909, 0.0052558
1: 0.0033579, 0.0069324, 0.0033532, 0.0069992, -0.0036412, 0.0035792
2: 0.0094620, 0.0192653, 0.0093304, 0.0192917, -0.0087137, 0.0088248
3: -0.0066698, -0.0023784, -0.0066704, -0.0023028, -0.0043670, 0.0042920
4: 0.0039045, 0.0053389, 0.0039043, 0.0053630, -0.0013481, 0.0013097
5: -0.0038266, -0.0005069, -0.0038424, -0.0004697, -0.0033569, 0.0033354
6: -0.0067280, -0.0051678, -0.0067292, -0.0051435, -0.0015845, 0.0015615
7: -0.0035290, -0.0005682, -0.0035950, -0.0005694, -0.0029596, 0.0030268
8: -0.0072268, -0.0007244, -0.0072456, -0.0006384, -0.0065884, 0.0065213
9: 1.0003859, 1.0015477, 1.0003765, 1.0016433, -0.0012574, 0.0011712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015203, upper bound: 0.0015199
time: 1.28 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015203, upper bound: 0.0015342
time: 1.16 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0047713, 0.0005286, -0.0046920, 0.0006092, -0.0053805, 0.0052207
1: 0.0033973, 0.0071079, 0.0033514, 0.0070309, -0.0036336, 0.0037565
2: 0.0091278, 0.0191372, 0.0092692, 0.0192927, -0.0090152, 0.0087613
3: -0.0066098, -0.0021823, -0.0066708, -0.0022665, -0.0043433, 0.0044886
4: 0.0039256, 0.0054017, 0.0039042, 0.0053746, -0.0013385, 0.0013546
5: -0.0037857, -0.0004224, -0.0038431, -0.0004534, -0.0033323, 0.0034208
6: -0.0067057, -0.0051054, -0.0067294, -0.0051322, -0.0015736, 0.0016239
7: -0.0037049, -0.0005937, -0.0036275, -0.0005668, -0.0031382, 0.0030338
8: -0.0071408, -0.0005088, -0.0072464, -0.0005987, -0.0065421, 0.0067376
9: 1.0003598, 1.0017896, 1.0003716, 1.0016876, -0.0013279, 0.0014180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015344, upper bound: 0.0016040
time: 1.37 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013393, upper bound: 0.0016567
time: 1.72 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045823, 0.0005985, -0.0048014, 0.0006476, -0.0052298, 0.0053999
1: 0.0033579, 0.0069324, 0.0033038, 0.0071209, -0.0037629, 0.0036286
2: 0.0094620, 0.0192653, 0.0090692, 0.0193512, -0.0087964, 0.0091052
3: -0.0066698, -0.0023784, -0.0067218, -0.0021633, -0.0045065, 0.0043434
4: 0.0039045, 0.0053389, 0.0038876, 0.0054085, -0.0013923, 0.0013222
5: -0.0038266, -0.0005069, -0.0038573, -0.0003970, -0.0034296, 0.0033503
6: -0.0067280, -0.0051678, -0.0067447, -0.0050971, -0.0016309, 0.0015770
7: -0.0035290, -0.0005682, -0.0036994, -0.0005165, -0.0030125, 0.0031312
8: -0.0072268, -0.0007244, -0.0072830, -0.0004692, -0.0067576, 0.0065587
9: 1.0003859, 1.0015477, 1.0003588, 1.0018258, -0.0014399, 0.0011889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015459, upper bound: 0.0015273
time: 1.54 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015459, upper bound: 0.0015390
time: 1.41 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0047713, 0.0005286, -0.0048370, 0.0006481, -0.0054194, 0.0053656
1: 0.0033973, 0.0071079, 0.0033019, 0.0071539, -0.0037567, 0.0038060
2: 0.0091278, 0.0191372, 0.0090055, 0.0193522, -0.0090881, 0.0090446
3: -0.0066098, -0.0021823, -0.0067221, -0.0021260, -0.0044838, 0.0045399
4: 0.0039256, 0.0054017, 0.0038875, 0.0054204, -0.0013829, 0.0013624
5: -0.0037857, -0.0004224, -0.0038581, -0.0003803, -0.0034054, 0.0034358
6: -0.0067057, -0.0051054, -0.0067448, -0.0050854, -0.0016203, 0.0016394
7: -0.0037049, -0.0005937, -0.0037320, -0.0005139, -0.0031910, 0.0031383
8: -0.0071408, -0.0005088, -0.0072838, -0.0004284, -0.0067124, 0.0067750
9: 1.0003598, 1.0017896, 1.0003542, 1.0018717, -0.0015119, 0.0014354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015617, upper bound: 0.0016116
time: 1.71 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015617, upper bound: 0.0016600
time: 1.71 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0050899, 0.0005009, -0.0046920, 0.0006092, -0.0056991, 0.0051929
1: 0.0033973, 0.0074080, 0.0033514, 0.0070309, -0.0036335, 0.0040567
2: 0.0085599, 0.0190878, 0.0092692, 0.0192927, -0.0096056, 0.0087356
3: -0.0065912, -0.0018426, -0.0066708, -0.0022665, -0.0043247, 0.0048283
4: 0.0039328, 0.0055110, 0.0039042, 0.0053746, -0.0013349, 0.0014640
5: -0.0037768, -0.0002731, -0.0038431, -0.0004534, -0.0033235, 0.0035700
6: -0.0066970, -0.0049991, -0.0067294, -0.0051322, -0.0015649, 0.0017303
7: -0.0040100, -0.0005884, -0.0036275, -0.0005668, -0.0034432, 0.0030391
8: -0.0071106, -0.0001411, -0.0072464, -0.0005987, -0.0065119, 0.0071053
9: 1.0003153, 1.0022029, 1.0003716, 1.0016876, -0.0013723, 0.0018313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012916, upper bound: 0.0013091
time: 1.56 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015096, upper bound: 0.0015989
time: 1.35 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0050899, 0.0005009, -0.0048370, 0.0006481, -0.0057380, 0.0053379
1: 0.0033973, 0.0074080, 0.0033019, 0.0071539, -0.0037566, 0.0041062
2: 0.0085599, 0.0190878, 0.0090055, 0.0193522, -0.0096785, 0.0090189
3: -0.0065912, -0.0018426, -0.0067221, -0.0021260, -0.0044652, 0.0048796
4: 0.0039328, 0.0055110, 0.0038875, 0.0054204, -0.0013794, 0.0014718
5: -0.0037768, -0.0002731, -0.0038581, -0.0003803, -0.0033965, 0.0035850
6: -0.0066970, -0.0049991, -0.0067448, -0.0050854, -0.0016116, 0.0017457
7: -0.0040100, -0.0005884, -0.0037320, -0.0005139, -0.0034961, 0.0031435
8: -0.0071106, -0.0001411, -0.0072838, -0.0004284, -0.0066822, 0.0071427
9: 1.0003153, 1.0022029, 1.0003542, 1.0018717, -0.0015564, 0.0018487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015304, upper bound: 0.0015427
time: 2.85 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015304, upper bound: 0.0016023
time: 1.44 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048278, 0.0005301, -0.0049481, 0.0005859, -0.0054137, 0.0054782
1: 0.0033941, 0.0071543, 0.0033546, 0.0072754, -0.0038812, 0.0037997
2: 0.0090253, 0.0191401, 0.0088181, 0.0192488, -0.0091244, 0.0092026
3: -0.0066109, -0.0021276, -0.0066533, -0.0019928, -0.0046180, 0.0045257
4: 0.0039253, 0.0054196, 0.0039111, 0.0054625, -0.0014254, 0.0013924
5: -0.0037876, -0.0003922, -0.0038333, -0.0003428, -0.0034448, 0.0034411
6: -0.0067061, -0.0050873, -0.0067225, -0.0050463, -0.0016598, 0.0016353
7: -0.0037476, -0.0005895, -0.0038787, -0.0005631, -0.0031845, 0.0032892
8: -0.0071428, -0.0004418, -0.0072188, -0.0003103, -0.0068325, 0.0067770
9: 1.0003535, 1.0018612, 1.0003350, 1.0020182, -0.0016648, 0.0015262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0015102
time: 1.42 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0016414
time: 2.24 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0050899, 0.0005009, -0.0050113, 0.0005875, -0.0056774, 0.0055122
1: 0.0033973, 0.0074080, 0.0033506, 0.0073251, -0.0039277, 0.0040574
2: 0.0085599, 0.0190878, 0.0087046, 0.0192520, -0.0094789, 0.0092155
3: -0.0065912, -0.0018426, -0.0066545, -0.0019347, -0.0046565, 0.0048119
4: 0.0039328, 0.0055110, 0.0039107, 0.0054817, -0.0013860, 0.0013987
5: -0.0037768, -0.0002731, -0.0038354, -0.0003103, -0.0034666, 0.0035623
6: -0.0066970, -0.0049991, -0.0067229, -0.0050264, -0.0016706, 0.0017238
7: -0.0040100, -0.0005884, -0.0039236, -0.0005583, -0.0034517, 0.0033352
8: -0.0071106, -0.0001411, -0.0072211, -0.0002366, -0.0068740, 0.0070799
9: 1.0003153, 1.0022029, 1.0003281, 1.0020975, -0.0017822, 0.0018748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014613, upper bound: 0.0015201
time: 1.39 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0012590, upper bound: 0.0015957
time: 1.78 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048278, 0.0005301, -0.0051156, 0.0006376, -0.0054654, 0.0056457
1: 0.0033941, 0.0071543, 0.0032984, 0.0074150, -0.0040209, 0.0038558
2: 0.0090253, 0.0191401, 0.0085128, 0.0193310, -0.0092434, 0.0095339
3: -0.0066109, -0.0021276, -0.0067119, -0.0018286, -0.0047822, 0.0045843
4: 0.0039253, 0.0054196, 0.0038911, 0.0055163, -0.0014748, 0.0014143
5: -0.0037876, -0.0003922, -0.0038553, -0.0002577, -0.0035299, 0.0034630
6: -0.0067061, -0.0050873, -0.0067413, -0.0049922, -0.0017139, 0.0016540
7: -0.0037476, -0.0005895, -0.0039943, -0.0005063, -0.0032413, 0.0034048
8: -0.0071428, -0.0004418, -0.0072682, -0.0001106, -0.0070322, 0.0068264
9: 1.0003535, 1.0018612, 1.0003155, 1.0022328, -0.0018793, 0.0015457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0015121
time: 1.56 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016432
time: 1.97 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0050899, 0.0005009, -0.0051742, 0.0006392, -0.0057291, 0.0056751
1: 0.0033973, 0.0074080, 0.0032942, 0.0074643, -0.0040670, 0.0041139
2: 0.0085599, 0.0190878, 0.0084066, 0.0193341, -0.0095826, 0.0095429
3: -0.0065912, -0.0018426, -0.0067131, -0.0017723, -0.0048188, 0.0048705
4: 0.0039328, 0.0055110, 0.0038907, 0.0055345, -0.0014334, 0.0014070
5: -0.0037768, -0.0002731, -0.0038575, -0.0002268, -0.0035500, 0.0035844
6: -0.0066970, -0.0049991, -0.0067417, -0.0049735, -0.0017236, 0.0017426
7: -0.0040100, -0.0005884, -0.0040409, -0.0005022, -0.0035078, 0.0034525
8: -0.0071106, -0.0001411, -0.0072705, -0.0000416, -0.0070690, 0.0071293
9: 1.0003153, 1.0022029, 1.0003084, 1.0023073, -0.0019920, 0.0018945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014664, upper bound: 0.0015249
time: 1.28 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014664, upper bound: 0.0015983
time: 1.30 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0049021, 0.0005535, -0.0046920, 0.0006092, -0.0055113, 0.0052455
1: 0.0033613, 0.0072162, 0.0033514, 0.0070309, -0.0036695, 0.0038648
2: 0.0088898, 0.0191780, 0.0092692, 0.0192927, -0.0092565, 0.0088056
3: -0.0066420, -0.0020554, -0.0066708, -0.0022665, -0.0043756, 0.0046155
4: 0.0039158, 0.0054432, 0.0039042, 0.0053746, -0.0013441, 0.0013888
5: -0.0037960, -0.0003543, -0.0038431, -0.0004534, -0.0033427, 0.0034888
6: -0.0067149, -0.0050634, -0.0067294, -0.0051322, -0.0015828, 0.0016660
7: -0.0038024, -0.0005529, -0.0036275, -0.0005668, -0.0032356, 0.0030746
8: -0.0071686, -0.0003537, -0.0072464, -0.0005987, -0.0065699, 0.0068927
9: 1.0003448, 1.0019563, 1.0003716, 1.0016876, -0.0013429, 0.0015848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013503, upper bound: 0.0016298
time: 1.67 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015384, upper bound: 0.0016935
time: 1.61 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0052395, 0.0005363, -0.0046920, 0.0006092, -0.0058487, 0.0052284
1: 0.0033517, 0.0075293, 0.0033514, 0.0070309, -0.0036791, 0.0041779
2: 0.0082920, 0.0191440, 0.0092692, 0.0192927, -0.0098835, 0.0088088
3: -0.0066337, -0.0017002, -0.0066708, -0.0022665, -0.0043673, 0.0049707
4: 0.0039195, 0.0055574, 0.0039042, 0.0053746, -0.0013494, 0.0015058
5: -0.0037931, -0.0001952, -0.0038431, -0.0004534, -0.0033397, 0.0036480
6: -0.0067099, -0.0049514, -0.0067294, -0.0051322, -0.0015778, 0.0017780
7: -0.0041169, -0.0005431, -0.0036275, -0.0005668, -0.0035502, 0.0030844
8: -0.0071470, 0.0000324, -0.0072464, -0.0005987, -0.0065484, 0.0072788
9: 1.0002981, 1.0023929, 1.0003716, 1.0016876, -0.0013895, 0.0020213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012958, upper bound: 0.0015411
time: 1.64 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012958, upper bound: 0.0014115
time: 3.49 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0049021, 0.0005535, -0.0048370, 0.0006481, -0.0055502, 0.0053905
1: 0.0033613, 0.0072162, 0.0033019, 0.0071539, -0.0037926, 0.0039143
2: 0.0088898, 0.0191780, 0.0090055, 0.0193522, -0.0092558, 0.0090136
3: -0.0066420, -0.0020554, -0.0067221, -0.0021260, -0.0045160, 0.0046668
4: 0.0039158, 0.0054432, 0.0038875, 0.0054204, -0.0013696, 0.0013810
5: -0.0037960, -0.0003543, -0.0038581, -0.0003803, -0.0034158, 0.0035038
6: -0.0067149, -0.0050634, -0.0067448, -0.0050854, -0.0016295, 0.0016815
7: -0.0038024, -0.0005529, -0.0037320, -0.0005139, -0.0032885, 0.0031791
8: -0.0071686, -0.0003537, -0.0072838, -0.0004284, -0.0067402, 0.0069302
9: 1.0003448, 1.0019563, 1.0003542, 1.0018717, -0.0015270, 0.0016022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015736, upper bound: 0.0016577
time: 1.93 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015736, upper bound: 0.0017066
time: 1.91 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0052395, 0.0005363, -0.0048370, 0.0006481, -0.0058876, 0.0053733
1: 0.0033517, 0.0075293, 0.0033019, 0.0071539, -0.0038022, 0.0042274
2: 0.0082920, 0.0191440, 0.0090055, 0.0193522, -0.0098775, 0.0090135
3: -0.0066337, -0.0017002, -0.0067221, -0.0021260, -0.0045077, 0.0050220
4: 0.0039195, 0.0055574, 0.0038875, 0.0054204, -0.0013675, 0.0014924
5: -0.0037931, -0.0001952, -0.0038581, -0.0003803, -0.0034128, 0.0036630
6: -0.0067099, -0.0049514, -0.0067448, -0.0050854, -0.0016245, 0.0017935
7: -0.0041169, -0.0005431, -0.0037320, -0.0005139, -0.0036031, 0.0031889
8: -0.0071470, 0.0000324, -0.0072838, -0.0004284, -0.0067187, 0.0073163
9: 1.0002981, 1.0023929, 1.0003542, 1.0018717, -0.0015736, 0.0020387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015376, upper bound: 0.0015577
time: 1.52 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013498, upper bound: 0.0016184
time: 1.58 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0049481, 0.0005859, -0.0055434, 0.0055031
1: 0.0033580, 0.0072629, 0.0033546, 0.0072754, -0.0039173, 0.0039083
2: 0.0087894, 0.0191808, 0.0088181, 0.0192488, -0.0093663, 0.0092402
3: -0.0066432, -0.0020018, -0.0066533, -0.0019928, -0.0046503, 0.0046514
4: 0.0039155, 0.0054603, 0.0039111, 0.0054625, -0.0014257, 0.0014289
5: -0.0037978, -0.0003250, -0.0038333, -0.0003428, -0.0034550, 0.0035083
6: -0.0067153, -0.0050458, -0.0067225, -0.0050463, -0.0016690, 0.0016767
7: -0.0038447, -0.0005489, -0.0038787, -0.0005631, -0.0032816, 0.0033297
8: -0.0071707, -0.0002886, -0.0072188, -0.0003103, -0.0068604, 0.0069302
9: 1.0003382, 1.0020260, 1.0003350, 1.0020182, -0.0016800, 0.0016910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015417, upper bound: 0.0015334
time: 1.42 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013155, upper bound: 0.0016726
time: 1.86 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0052395, 0.0005363, -0.0050113, 0.0005875, -0.0058270, 0.0055476
1: 0.0033517, 0.0075293, 0.0033506, 0.0073251, -0.0039734, 0.0041786
2: 0.0082920, 0.0191440, 0.0087046, 0.0192520, -0.0097504, 0.0092800
3: -0.0066337, -0.0017002, -0.0066545, -0.0019347, -0.0046990, 0.0049543
4: 0.0039195, 0.0055574, 0.0039107, 0.0054817, -0.0013919, 0.0014356
5: -0.0037931, -0.0001952, -0.0038354, -0.0003103, -0.0034828, 0.0036402
6: -0.0067099, -0.0049514, -0.0067229, -0.0050264, -0.0016835, 0.0017715
7: -0.0041169, -0.0005431, -0.0039236, -0.0005583, -0.0035586, 0.0033805
8: -0.0071470, 0.0000324, -0.0072211, -0.0002366, -0.0069105, 0.0072535
9: 1.0002981, 1.0023929, 1.0003281, 1.0020975, -0.0017993, 0.0020648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014649, upper bound: 0.0015279
time: 2.05 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014649, upper bound: 0.0016091
time: 1.79 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0051156, 0.0006376, -0.0055951, 0.0056705
1: 0.0033580, 0.0072629, 0.0032984, 0.0074150, -0.0040570, 0.0039644
2: 0.0087894, 0.0191808, 0.0085128, 0.0193310, -0.0094006, 0.0094901
3: -0.0066432, -0.0020018, -0.0067119, -0.0018286, -0.0048145, 0.0047101
4: 0.0039155, 0.0054603, 0.0038911, 0.0055163, -0.0014540, 0.0014242
5: -0.0037978, -0.0003250, -0.0038553, -0.0002577, -0.0035401, 0.0035302
6: -0.0067153, -0.0050458, -0.0067413, -0.0049922, -0.0017231, 0.0016954
7: -0.0038447, -0.0005489, -0.0039943, -0.0005063, -0.0033383, 0.0034454
8: -0.0071707, -0.0002886, -0.0072682, -0.0001106, -0.0070601, 0.0069796
9: 1.0003382, 1.0020260, 1.0003155, 1.0022328, -0.0018946, 0.0017104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015571, upper bound: 0.0015418
time: 1.38 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015571, upper bound: 0.0016829
time: 1.78 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0052395, 0.0005363, -0.0051742, 0.0006392, -0.0058787, 0.0057105
1: 0.0033517, 0.0075293, 0.0032942, 0.0074643, -0.0041126, 0.0042351
2: 0.0082920, 0.0191440, 0.0084066, 0.0193341, -0.0097668, 0.0095241
3: -0.0066337, -0.0017002, -0.0067131, -0.0017723, -0.0048614, 0.0050129
4: 0.0039195, 0.0055574, 0.0038907, 0.0055345, -0.0014199, 0.0014285
5: -0.0037931, -0.0001952, -0.0038575, -0.0002268, -0.0035663, 0.0036623
6: -0.0067099, -0.0049514, -0.0067417, -0.0049735, -0.0017365, 0.0017903
7: -0.0041169, -0.0005431, -0.0040409, -0.0005022, -0.0036147, 0.0034978
8: -0.0071470, 0.0000324, -0.0072705, -0.0000416, -0.0071055, 0.0073029
9: 1.0002981, 1.0023929, 1.0003084, 1.0023073, -0.0020092, 0.0020845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012656, upper bound: 0.0015416
time: 1.68 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014744, upper bound: 0.0016152
time: 1.53 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.56 seconds
IS_B1_A1_A2_B2_A1_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015206, upper bound: 0.0015207
IS_B1_A1_A2_B2_A1_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015206, upper bound: 0.0015349
IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016040
IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016569
IS_B1_A1_A2_B2_A1_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015461, upper bound: 0.0015279
IS_B1_A1_A2_B2_A1_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015461, upper bound: 0.0015279
IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0016259, upper bound: 0.0015397
IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0016259, upper bound: 0.0015397
IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0015112
IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0016414
IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015401, upper bound: 0.0015133
IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015401, upper bound: 0.0016433
IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015109, upper bound: 0.0015349
IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015109, upper bound: 0.0015992
IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014622, upper bound: 0.0015208
IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014622, upper bound: 0.0015962
IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0015435
IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0016027
IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0014659
IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0015987
IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016298
IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016935
IS_B1_A1_A2_B2_A2_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
IS_B1_A1_A2_B2_A2_A1_B1_B2_B2_A2, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
IS_B1_A1_A2_B2_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0016576
IS_B1_A1_A2_B2_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0017067
IS_B1_A1_A2_B2_A2_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014681, upper bound: 0.0015151
IS_B1_A1_A2_B2_A2_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014681, upper bound: 0.0015151
IS_B1_A1_A2_B2_A2_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0013112, upper bound: 0.0015418
IS_B1_A1_A2_B2_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0016829
IS_B1_A1_A2_B2_A2_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0015413
IS_B1_A1_A2_B2_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0016119
IS_B1_A1_A2_B2_A2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0015281
IS_B1_A1_A2_B2_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0016091
IS_B1_A1_A2_B2_A2_A2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0015579
IS_B1_A1_A2_B2_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0016184
IS_B1_A1_A2_B2_A2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0015417
IS_B1_A1_A2_B2_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0016152
IS_B1_A2_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015203, upper bound: 0.0015199
IS_B1_A2_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015203, upper bound: 0.0015342
IS_B1_A2_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015344, upper bound: 0.0016040
IS_B1_A2_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0013393, upper bound: 0.0016567
IS_B1_A2_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015459, upper bound: 0.0015273
IS_B1_A2_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015459, upper bound: 0.0015390
IS_B1_A2_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015617, upper bound: 0.0016116
IS_B1_A2_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015617, upper bound: 0.0016600
IS_B1_A2_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0012916, upper bound: 0.0013091
IS_B1_A2_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015096, upper bound: 0.0015989
IS_B1_A2_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015304, upper bound: 0.0015427
IS_B1_A2_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015304, upper bound: 0.0016023
IS_B1_A2_A2_B2_A1_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0015102
IS_B1_A2_A2_B2_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0016414
IS_B1_A2_A2_B2_A1_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014613, upper bound: 0.0015201
IS_B1_A2_A2_B2_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0012590, upper bound: 0.0015957
IS_B1_A2_A2_B2_A1_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0015121
IS_B1_A2_A2_B2_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016432
IS_B1_A2_A2_B2_A1_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014664, upper bound: 0.0015249
IS_B1_A2_A2_B2_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014664, upper bound: 0.0015983
IS_B1_A2_A2_B2_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0013503, upper bound: 0.0016298
IS_B1_A2_A2_B2_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015384, upper bound: 0.0016935
IS_B1_A2_A2_B2_A2_B1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0012958, upper bound: 0.0015411
IS_B1_A2_A2_B2_A2_B1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0012958, upper bound: 0.0014115
IS_B1_A2_A2_B2_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015736, upper bound: 0.0016577
IS_B1_A2_A2_B2_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015736, upper bound: 0.0017066
IS_B1_A2_A2_B2_A2_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015376, upper bound: 0.0015577
IS_B1_A2_A2_B2_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0013498, upper bound: 0.0016184
IS_B1_A2_A2_B2_A2_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015417, upper bound: 0.0015334
IS_B1_A2_A2_B2_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0013155, upper bound: 0.0016726
IS_B1_A2_A2_B2_A2_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014649, upper bound: 0.0015279
IS_B1_A2_A2_B2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014649, upper bound: 0.0016091
IS_B1_A2_A2_B2_A2_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015571, upper bound: 0.0015418
IS_B1_A2_A2_B2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0015571, upper bound: 0.0016829
IS_B1_A2_A2_B2_A2_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0012656, upper bound: 0.0015416
IS_B1_A2_A2_B2_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.56
Output dim: 9, lower bound: -0.0014744, upper bound: 0.0016152

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0044601, 0.0006993, -0.0053299, 0.0050681
1: 0.0033547, 0.0069801, 0.0032986, 0.0068191, -0.0034644, 0.0036815
2: 0.0093798, 0.0192902, 0.0096808, 0.0194604, -0.0089184, 0.0084605
3: -0.0066698, -0.0023249, -0.0067452, -0.0025091, -0.0041607, 0.0044203
4: 0.0039045, 0.0053556, 0.0038797, 0.0052965, -0.0012453, 0.0013261
5: -0.0038415, -0.0004855, -0.0038883, -0.0005638, -0.0032777, 0.0034028
6: -0.0067290, -0.0051519, -0.0067565, -0.0052091, -0.0015200, 0.0016046
7: -0.0035825, -0.0005714, -0.0034129, -0.0005372, -0.0030454, 0.0028415
8: -0.0072445, -0.0006709, -0.0073556, -0.0008654, -0.0063792, 0.0066847
9: 1.0003786, 1.0016094, 1.0004028, 1.0013897, -0.0010111, 0.0012066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0011491, upper bound: 0.0010796
time: 1.85 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015297, upper bound: 0.0015984
time: 1.27 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0046305, 0.0006080, -0.0052385, 0.0052385
1: 0.0033547, 0.0069801, 0.0033547, 0.0069801, -0.0036254, 0.0036254
2: 0.0093798, 0.0192902, 0.0093798, 0.0192902, -0.0087331, 0.0087331
3: -0.0066698, -0.0023249, -0.0066698, -0.0023249, -0.0043449, 0.0043449
4: 0.0039045, 0.0053556, 0.0039045, 0.0053556, -0.0012900, 0.0012900
5: -0.0038415, -0.0004855, -0.0038415, -0.0004855, -0.0033560, 0.0033560
6: -0.0067290, -0.0051519, -0.0067290, -0.0051519, -0.0015772, 0.0015772
7: -0.0035825, -0.0005714, -0.0035825, -0.0005714, -0.0030111, 0.0030111
8: -0.0072445, -0.0006709, -0.0072445, -0.0006709, -0.0065736, 0.0065736
9: 1.0003786, 1.0016094, 1.0003786, 1.0016094, -0.0012308, 0.0012308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0011491, upper bound: 0.0011842
time: 1.64 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015297, upper bound: 0.0016516
time: 1.62 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0044601, 0.0006993, -0.0047777, 0.0006468, -0.0051069, 0.0054770
1: 0.0032986, 0.0068191, 0.0033051, 0.0071064, -0.0038078, 0.0035140
2: 0.0096808, 0.0194604, 0.0091127, 0.0193497, -0.0085433, 0.0092064
3: -0.0067452, -0.0025091, -0.0067212, -0.0021812, -0.0045640, 0.0042121
4: 0.0038797, 0.0052965, 0.0038877, 0.0054023, -0.0013708, 0.0012578
5: -0.0038883, -0.0005638, -0.0038564, -0.0004109, -0.0034774, 0.0032926
6: -0.0067565, -0.0052091, -0.0067445, -0.0051042, -0.0016523, 0.0015355
7: -0.0034129, -0.0005372, -0.0036876, -0.0005181, -0.0028948, 0.0031504
8: -0.0073556, -0.0008654, -0.0072819, -0.0004982, -0.0068574, 0.0064166
9: 1.0004028, 1.0013897, 1.0003608, 1.0017968, -0.0013940, 0.0010289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0009458, upper bound: 0.0012159
time: 1.16 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015406, upper bound: 0.0015347
time: 1.21 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0047777, 0.0006468, -0.0052774, 0.0053856
1: 0.0033547, 0.0069801, 0.0033051, 0.0071064, -0.0037517, 0.0036750
2: 0.0093798, 0.0192902, 0.0091127, 0.0193497, -0.0088063, 0.0090156
3: -0.0066698, -0.0023249, -0.0067212, -0.0021812, -0.0044886, 0.0043964
4: 0.0039045, 0.0053556, 0.0038877, 0.0054023, -0.0013342, 0.0012978
5: -0.0038415, -0.0004855, -0.0038564, -0.0004109, -0.0034306, 0.0033709
6: -0.0067290, -0.0051519, -0.0067445, -0.0051042, -0.0016248, 0.0015927
7: -0.0035825, -0.0005714, -0.0036876, -0.0005181, -0.0030644, 0.0031162
8: -0.0072445, -0.0006709, -0.0072819, -0.0004982, -0.0067464, 0.0066110
9: 1.0003786, 1.0016094, 1.0003608, 1.0017968, -0.0014182, 0.0012486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0009458, upper bound: 0.0014021
time: 1.23 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015406, upper bound: 0.0015230
time: 1.23 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0049481, 0.0005859, -0.0052164, 0.0055561
1: 0.0033547, 0.0069801, 0.0033546, 0.0072754, -0.0039206, 0.0036256
2: 0.0093798, 0.0192902, 0.0088181, 0.0192488, -0.0087071, 0.0093153
3: -0.0066698, -0.0023249, -0.0066533, -0.0019928, -0.0046770, 0.0043284
4: 0.0039045, 0.0053556, 0.0039111, 0.0054625, -0.0014084, 0.0012956
5: -0.0038415, -0.0004855, -0.0038333, -0.0003428, -0.0034987, 0.0033478
6: -0.0067290, -0.0051519, -0.0067225, -0.0050463, -0.0016827, 0.0015706
7: -0.0035825, -0.0005714, -0.0038787, -0.0005631, -0.0030194, 0.0033073
8: -0.0072445, -0.0006709, -0.0072188, -0.0003103, -0.0069342, 0.0065478
9: 1.0003786, 1.0016094, 1.0003350, 1.0020182, -0.0016396, 0.0012745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014058, upper bound: 0.0016040
time: 2.15 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013619, upper bound: 0.0015928
time: 1.68 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0051156, 0.0006376, -0.0052681, 0.0057236
1: 0.0033547, 0.0069801, 0.0032984, 0.0074150, -0.0040603, 0.0036817
2: 0.0093798, 0.0192902, 0.0085128, 0.0193310, -0.0088199, 0.0096466
3: -0.0066698, -0.0023249, -0.0067119, -0.0018286, -0.0048412, 0.0043870
4: 0.0039045, 0.0053556, 0.0038911, 0.0055163, -0.0014578, 0.0013147
5: -0.0038415, -0.0004855, -0.0038553, -0.0002577, -0.0035838, 0.0033698
6: -0.0067290, -0.0051519, -0.0067413, -0.0049922, -0.0017368, 0.0015894
7: -0.0035825, -0.0005714, -0.0039943, -0.0005063, -0.0030762, 0.0034229
8: -0.0072445, -0.0006709, -0.0072682, -0.0001106, -0.0071339, 0.0065972
9: 1.0003786, 1.0016094, 1.0003155, 1.0022328, -0.0018542, 0.0012939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 212

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014080, upper bound: 0.0016055
time: 1.90 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013646, upper bound: 0.0015946
time: 1.52 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0046305, 0.0006080, -0.0055561, 0.0052164
1: 0.0033546, 0.0072754, 0.0033547, 0.0069801, -0.0036256, 0.0039206
2: 0.0088181, 0.0192488, 0.0093798, 0.0192902, -0.0093153, 0.0087071
3: -0.0066533, -0.0019928, -0.0066698, -0.0023249, -0.0043284, 0.0046770
4: 0.0039111, 0.0054625, 0.0039045, 0.0053556, -0.0012956, 0.0014084
5: -0.0038333, -0.0003428, -0.0038415, -0.0004855, -0.0033478, 0.0034987
6: -0.0067225, -0.0050463, -0.0067290, -0.0051519, -0.0015706, 0.0016827
7: -0.0038787, -0.0005631, -0.0035825, -0.0005714, -0.0033073, 0.0030194
8: -0.0072188, -0.0003103, -0.0072445, -0.0006709, -0.0065478, 0.0069342
9: 1.0003350, 1.0020182, 1.0003786, 1.0016094, -0.0012745, 0.0016396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0010356, upper bound: 0.0010529
time: 1.48 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015059, upper bound: 0.0015933
time: 1.34 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0049481, 0.0005859, -0.0055340, 0.0055340
1: 0.0033546, 0.0072754, 0.0033546, 0.0072754, -0.0039208, 0.0039208
2: 0.0088181, 0.0192488, 0.0088181, 0.0192488, -0.0091842, 0.0091842
3: -0.0066533, -0.0019928, -0.0066533, -0.0019928, -0.0046604, 0.0046604
4: 0.0039111, 0.0054625, 0.0039111, 0.0054625, -0.0013371, 0.0013371
5: -0.0038333, -0.0003428, -0.0038333, -0.0003428, -0.0034905, 0.0034905
6: -0.0067225, -0.0050463, -0.0067225, -0.0050463, -0.0016762, 0.0016762
7: -0.0038787, -0.0005631, -0.0038787, -0.0005631, -0.0033156, 0.0033156
8: -0.0072188, -0.0003103, -0.0072188, -0.0003103, -0.0069084, 0.0069084
9: 1.0003350, 1.0020182, 1.0003350, 1.0020182, -0.0016832, 0.0016832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0010081, upper bound: 0.0010409
time: 1.38 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014567, upper bound: 0.0015146
time: 1.20 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0047777, 0.0006468, -0.0055950, 0.0053636
1: 0.0033546, 0.0072754, 0.0033051, 0.0071064, -0.0037518, 0.0039702
2: 0.0088181, 0.0192488, 0.0091127, 0.0193497, -0.0093885, 0.0089896
3: -0.0066533, -0.0019928, -0.0067212, -0.0021812, -0.0044720, 0.0047284
4: 0.0039111, 0.0054625, 0.0038877, 0.0054023, -0.0013398, 0.0014162
5: -0.0038333, -0.0003428, -0.0038564, -0.0004109, -0.0034224, 0.0035135
6: -0.0067225, -0.0050463, -0.0067445, -0.0051042, -0.0016183, 0.0016982
7: -0.0038787, -0.0005631, -0.0036876, -0.0005181, -0.0033605, 0.0031245
8: -0.0072188, -0.0003103, -0.0072819, -0.0004982, -0.0067206, 0.0069716
9: 1.0003350, 1.0020182, 1.0003608, 1.0017968, -0.0014619, 0.0016574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0009403, upper bound: 0.0013277
time: 1.48 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015267, upper bound: 0.0015968
time: 1.81 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0051156, 0.0006376, -0.0055857, 0.0057015
1: 0.0033546, 0.0072754, 0.0032984, 0.0074150, -0.0040604, 0.0039769
2: 0.0088181, 0.0192488, 0.0085128, 0.0193310, -0.0092880, 0.0095108
3: -0.0066533, -0.0019928, -0.0067119, -0.0018286, -0.0048246, 0.0047190
4: 0.0039111, 0.0054625, 0.0038911, 0.0055163, -0.0013833, 0.0013454
5: -0.0038333, -0.0003428, -0.0038553, -0.0002577, -0.0035756, 0.0035124
6: -0.0067225, -0.0050463, -0.0067413, -0.0049922, -0.0017303, 0.0016949
7: -0.0038787, -0.0005631, -0.0039943, -0.0005063, -0.0033723, 0.0034312
8: -0.0072188, -0.0003103, -0.0072682, -0.0001106, -0.0071082, 0.0069578
9: 1.0003350, 1.0020182, 1.0003155, 1.0022328, -0.0018978, 0.0017027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014257, upper bound: 0.0015762
time: 1.99 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014237, upper bound: 0.0015741
time: 1.67 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0044601, 0.0006993, -0.0054770, 0.0051069
1: 0.0033051, 0.0071064, 0.0032986, 0.0068191, -0.0035140, 0.0038078
2: 0.0091127, 0.0193497, 0.0096808, 0.0194604, -0.0092064, 0.0085433
3: -0.0067212, -0.0021812, -0.0067452, -0.0025091, -0.0042121, 0.0045640
4: 0.0038877, 0.0054023, 0.0038797, 0.0052965, -0.0012578, 0.0013708
5: -0.0038564, -0.0004109, -0.0038883, -0.0005638, -0.0032926, 0.0034774
6: -0.0067445, -0.0051042, -0.0067565, -0.0052091, -0.0015355, 0.0016523
7: -0.0036876, -0.0005181, -0.0034129, -0.0005372, -0.0031504, 0.0028948
8: -0.0072819, -0.0004982, -0.0073556, -0.0008654, -0.0064166, 0.0068574
9: 1.0003608, 1.0017968, 1.0004028, 1.0013897, -0.0010289, 0.0013940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012007, upper bound: 0.0011606
time: 1.51 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0016232
time: 1.20 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 6.08 seconds
IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0011491, upper bound: 0.0010796
IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0015297, upper bound: 0.0015984
IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0011491, upper bound: 0.0011842
IS_B1_A1_A2_B2_A1_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0015297, upper bound: 0.0016516
IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0009458, upper bound: 0.0012159
IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0015406, upper bound: 0.0015347
IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0009458, upper bound: 0.0014021
IS_B1_A1_A2_B2_A1_A1_B1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0015406, upper bound: 0.0015230
IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0014058, upper bound: 0.0016040
IS_B1_A1_A2_B2_A1_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0013619, upper bound: 0.0015928
IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0014080, upper bound: 0.0016055
IS_B1_A1_A2_B2_A1_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0013646, upper bound: 0.0015946
IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0010356, upper bound: 0.0010529
IS_B1_A1_A2_B2_A1_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0015059, upper bound: 0.0015933
IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0010081, upper bound: 0.0010409
IS_B1_A1_A2_B2_A1_A2_B1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0014567, upper bound: 0.0015146
IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0009403, upper bound: 0.0013277
IS_B1_A1_A2_B2_A1_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0015267, upper bound: 0.0015968
IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0014257, upper bound: 0.0015762
IS_B1_A1_A2_B2_A1_A2_B2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0014237, upper bound: 0.0015741
IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0012007, upper bound: 0.0011606
IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 6.08
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0016232
IS_B1_A1_A2_B2_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016935
IS_B1_A1_A2_B2_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0016576
IS_B1_A1_A2_B2_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0017067
IS_B1_A1_A2_B2_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0016829
IS_B1_A1_A2_B2_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0016119
IS_B1_A1_A2_B2_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0016091
IS_B1_A1_A2_B2_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0016184
IS_B1_A1_A2_B2_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0016152
IS_B1_A2_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015344, upper bound: 0.0016040
IS_B1_A2_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0013393, upper bound: 0.0016567
IS_B1_A2_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015617, upper bound: 0.0016116
IS_B1_A2_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015617, upper bound: 0.0016600
IS_B1_A2_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015096, upper bound: 0.0015989
IS_B1_A2_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015304, upper bound: 0.0016023
IS_B1_A2_A2_B2_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0016414
IS_B1_A2_A2_B2_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0012590, upper bound: 0.0015957
IS_B1_A2_A2_B2_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016432
IS_B1_A2_A2_B2_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0014664, upper bound: 0.0015983
IS_B1_A2_A2_B2_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0013503, upper bound: 0.0016298
IS_B1_A2_A2_B2_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015384, upper bound: 0.0016935
IS_B1_A2_A2_B2_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015736, upper bound: 0.0016577
IS_B1_A2_A2_B2_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015736, upper bound: 0.0017066
IS_B1_A2_A2_B2_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0013498, upper bound: 0.0016184
IS_B1_A2_A2_B2_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0013155, upper bound: 0.0016726
IS_B1_A2_A2_B2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0014649, upper bound: 0.0016091
IS_B1_A2_A2_B2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0015571, upper bound: 0.0016829
IS_B1_A2_A2_B2_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.08
Output dim: 9, lower bound: -0.0014744, upper bound: 0.0016152

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.89 + 598.04 = 601.93 seconds
