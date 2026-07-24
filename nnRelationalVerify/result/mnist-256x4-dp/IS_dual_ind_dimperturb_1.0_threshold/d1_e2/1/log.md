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
execution time: IAR + RelationalAnalysis = 1.14 + 2.71 = 3.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0017606, upper bound: 0.0017606

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017606, upper bound: 0.0017596
time: 1.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
time: 1.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.14 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.14
Output dim: 9, lower bound: -0.0017606, upper bound: 0.0017596
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.14
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0050730, 0.0008325, -0.0050904, 0.0010837, -0.0061567, 0.0059229
1: 0.0031296, 0.0073050, 0.0029200, 0.0073178, -0.0041881, 0.0043850
2: 0.0085573, 0.0196800, 0.0085237, 0.0201408, -0.0106694, 0.0102578
3: -0.0069078, -0.0019382, -0.0071492, -0.0019233, -0.0049845, 0.0052110
4: 0.0038273, 0.0054838, 0.0037488, 0.0054888, -0.0015904, 0.0016540
5: -0.0039506, -0.0002278, -0.0040829, -0.0002159, -0.0037348, 0.0038551
6: -0.0068052, -0.0050149, -0.0068849, -0.0050096, -0.0017956, 0.0018701
7: -0.0038496, -0.0003388, -0.0038616, -0.0001440, -0.0037056, 0.0035228
8: -0.0074959, -0.0001297, -0.0077969, -0.0001072, -0.0073886, 0.0076673
9: 1.0003341, 1.0021654, 1.0003322, 1.0021871, -0.0018530, 0.0018332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
time: 1.49 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
time: 1.49 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0052048, 0.0007401, -0.0050885, 0.0010356, -0.0062404, 0.0058287
1: 0.0031823, 0.0074221, 0.0029572, 0.0073165, -0.0041342, 0.0044649
2: 0.0083194, 0.0195125, 0.0085274, 0.0200529, -0.0108410, 0.0101865
3: -0.0068315, -0.0018053, -0.0071050, -0.0019249, -0.0049066, 0.0052997
4: 0.0038542, 0.0055269, 0.0037636, 0.0054883, -0.0016340, 0.0016979
5: -0.0038910, -0.0001663, -0.0040548, -0.0002172, -0.0036738, 0.0038885
6: -0.0067760, -0.0049719, -0.0068698, -0.0050102, -0.0017658, 0.0018979
7: -0.0039687, -0.0003721, -0.0038603, -0.0001770, -0.0037918, 0.0034882
8: -0.0073862, 0.0000242, -0.0077394, -0.0001097, -0.0072765, 0.0077636
9: 1.0003170, 1.0023339, 1.0003325, 1.0021849, -0.0018679, 0.0020014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016981
time: 1.68 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017419, upper bound: 0.0017419
time: 1.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.58 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.58
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.58
Output dim: 9, lower bound: -0.0017596, upper bound: 0.0017596
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.58
Output dim: 9, lower bound: -0.0015626, upper bound: 0.0016981
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.58
Output dim: 9, lower bound: -0.0017419, upper bound: 0.0017419

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016988, upper bound: 0.0015626
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017429, upper bound: 0.0017420
time: 1.64 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0050730, 0.0008325, -0.0052048, 0.0007401, -0.0058131, 0.0060373
1: 0.0031296, 0.0073050, 0.0031823, 0.0074221, -0.0042924, 0.0041227
2: 0.0085573, 0.0196800, 0.0083194, 0.0195125, -0.0100740, 0.0104818
3: -0.0069078, -0.0019382, -0.0068315, -0.0018053, -0.0051024, 0.0048933
4: 0.0038273, 0.0054838, 0.0038542, 0.0055269, -0.0016261, 0.0015659
5: -0.0039506, -0.0002278, -0.0038910, -0.0001663, -0.0037843, 0.0036632
6: -0.0068052, -0.0050149, -0.0067760, -0.0049719, -0.0018333, 0.0017612
7: -0.0038496, -0.0003388, -0.0039687, -0.0003721, -0.0034775, 0.0036299
8: -0.0074959, -0.0001297, -0.0073862, 0.0000242, -0.0075201, 0.0072566
9: 1.0003341, 1.0021654, 1.0003170, 1.0023339, -0.0019997, 0.0018485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016988, upper bound: 0.0015626
time: 1.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0017430, upper bound: 0.0017420
time: 1.90 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0051472, 0.0007367, -0.0046109, 0.0008954, -0.0060426, 0.0053476
1: 0.0031881, 0.0073828, 0.0030708, 0.0069910, -0.0038029, 0.0043120
2: 0.0084274, 0.0195058, 0.0094224, 0.0197907, -0.0104496, 0.0092876
3: -0.0068286, -0.0018535, -0.0069822, -0.0023225, -0.0045061, 0.0051287
4: 0.0038551, 0.0055108, 0.0038045, 0.0053549, -0.0014998, 0.0016558
5: -0.0038880, -0.0002017, -0.0039614, -0.0005081, -0.0033799, 0.0037598
6: -0.0067750, -0.0049895, -0.0068264, -0.0051556, -0.0016194, 0.0018370
7: -0.0039401, -0.0003782, -0.0036248, -0.0002613, -0.0036787, 0.0032466
8: -0.0073816, -0.0000473, -0.0075653, -0.0007012, -0.0066804, 0.0075181
9: 1.0003222, 1.0022615, 1.0003752, 1.0015893, -0.0012671, 0.0018864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0016981
time: 1.23 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0052048, 0.0007401, -0.0049922, 0.0010257, -0.0062305, 0.0057323
1: 0.0031823, 0.0074221, 0.0029684, 0.0072537, -0.0040714, 0.0044536
2: 0.0083194, 0.0195125, 0.0087093, 0.0200340, -0.0108222, 0.0097774
3: -0.0068315, -0.0018053, -0.0070967, -0.0020013, -0.0048302, 0.0052914
4: 0.0038542, 0.0055269, 0.0037663, 0.0054627, -0.0015393, 0.0016963
5: -0.0038910, -0.0001663, -0.0040472, -0.0002790, -0.0036120, 0.0038809
6: -0.0067760, -0.0049719, -0.0068669, -0.0050389, -0.0017371, 0.0018950
7: -0.0039687, -0.0003721, -0.0038119, -0.0001907, -0.0037780, 0.0034398
8: -0.0073862, 0.0000242, -0.0077266, -0.0002309, -0.0071553, 0.0077508
9: 1.0003170, 1.0023339, 1.0003407, 1.0020653, -0.0017483, 0.0019932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016981, upper bound: 0.0015626
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016981, upper bound: 0.0017419
time: 1.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.24 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 9, lower bound: -0.0016988, upper bound: 0.0015626
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 9, lower bound: -0.0017429, upper bound: 0.0017420
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 9, lower bound: -0.0016988, upper bound: 0.0015626
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 9, lower bound: -0.0017430, upper bound: 0.0017420
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.24
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0016981
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 9, lower bound: -0.0016981, upper bound: 0.0015626
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.24
Output dim: 9, lower bound: -0.0016981, upper bound: 0.0017419

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015587, upper bound: 0.0015588
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015587, upper bound: 0.0015632
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015632, upper bound: 0.0016989
time: 1.55 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015632, upper bound: 0.0017430
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0045952, 0.0006857, -0.0051472, 0.0007367, -0.0053319, 0.0058329
1: 0.0032486, 0.0069791, 0.0031881, 0.0073828, -0.0041342, 0.0037910
2: 0.0094526, 0.0194093, 0.0084274, 0.0195058, -0.0091744, 0.0100747
3: -0.0067785, -0.0023363, -0.0068286, -0.0018535, -0.0049249, 0.0044923
4: 0.0038708, 0.0053503, 0.0038551, 0.0055108, -0.0015780, 0.0014422
5: -0.0038524, -0.0005188, -0.0038880, -0.0002017, -0.0036507, 0.0033692
6: -0.0067591, -0.0051604, -0.0067750, -0.0049895, -0.0017696, 0.0016146
7: -0.0036138, -0.0004292, -0.0039401, -0.0003782, -0.0032356, 0.0035109
8: -0.0073157, -0.0007215, -0.0073816, -0.0000473, -0.0072684, 0.0066602
9: 1.0003768, 1.0015694, 1.0003222, 1.0022615, -0.0018847, 0.0012472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015585, upper bound: 0.0015582
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015585, upper bound: 0.0015626
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049766, 0.0008226, -0.0052048, 0.0007401, -0.0057167, 0.0060274
1: 0.0031411, 0.0072423, 0.0031823, 0.0074221, -0.0042810, 0.0040600
2: 0.0087394, 0.0196612, 0.0083194, 0.0195125, -0.0096843, 0.0104633
3: -0.0068998, -0.0020146, -0.0068315, -0.0018053, -0.0050945, 0.0048169
4: 0.0038299, 0.0054582, 0.0038542, 0.0055269, -0.0016245, 0.0014876
5: -0.0039430, -0.0002896, -0.0038910, -0.0001663, -0.0037767, 0.0036014
6: -0.0068023, -0.0050436, -0.0067760, -0.0049719, -0.0018304, 0.0017324
7: -0.0038014, -0.0003530, -0.0039687, -0.0003721, -0.0034292, 0.0036157
8: -0.0074830, -0.0002510, -0.0073862, 0.0000242, -0.0075072, 0.0071352
9: 1.0003421, 1.0020459, 1.0003170, 1.0023339, -0.0019917, 0.0017289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015630, upper bound: 0.0016981
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015630, upper bound: 0.0017420
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0051029, 0.0007293, -0.0046109, 0.0008954, -0.0059983, 0.0053402
1: 0.0031957, 0.0073574, 0.0030708, 0.0069910, -0.0037953, 0.0042866
2: 0.0085136, 0.0194918, 0.0094224, 0.0197907, -0.0103736, 0.0092742
3: -0.0068226, -0.0018850, -0.0069822, -0.0023225, -0.0045000, 0.0050973
4: 0.0038572, 0.0054998, 0.0038045, 0.0053549, -0.0014977, 0.0016399
5: -0.0038828, -0.0002325, -0.0039614, -0.0005081, -0.0033747, 0.0037290
6: -0.0067728, -0.0050023, -0.0068264, -0.0051556, -0.0016172, 0.0018241
7: -0.0039171, -0.0003870, -0.0036248, -0.0002613, -0.0036557, 0.0032378
8: -0.0073722, -0.0001054, -0.0075653, -0.0007012, -0.0066709, 0.0074599
9: 1.0003257, 1.0022068, 1.0003752, 1.0015893, -0.0012636, 0.0018317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0016981
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0016981
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0047391, 0.0005960, -0.0049922, 0.0010257, -0.0057648, 0.0055881
1: 0.0032949, 0.0071035, 0.0029684, 0.0072537, -0.0039588, 0.0041351
2: 0.0091941, 0.0192440, 0.0087093, 0.0200340, -0.0099472, 0.0097166
3: -0.0067071, -0.0021938, -0.0070967, -0.0020013, -0.0047058, 0.0049029
4: 0.0038956, 0.0053966, 0.0037663, 0.0054627, -0.0015671, 0.0015760
5: -0.0037981, -0.0004521, -0.0040472, -0.0002790, -0.0035191, 0.0035952
6: -0.0067318, -0.0051139, -0.0068669, -0.0050389, -0.0016930, 0.0017530
7: -0.0037385, -0.0004570, -0.0038119, -0.0001907, -0.0035477, 0.0033549
8: -0.0072069, -0.0005544, -0.0077266, -0.0002309, -0.0069760, 0.0071722
9: 1.0003587, 1.0017505, 1.0003407, 1.0020653, -0.0017066, 0.0014098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015626
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0051029, 0.0007293, -0.0049922, 0.0010257, -0.0061286, 0.0057215
1: 0.0031957, 0.0073574, 0.0029684, 0.0072537, -0.0040580, 0.0043890
2: 0.0085136, 0.0194918, 0.0087093, 0.0200340, -0.0104296, 0.0097566
3: -0.0068226, -0.0018850, -0.0070967, -0.0020013, -0.0048213, 0.0052118
4: 0.0038572, 0.0054998, 0.0037663, 0.0054627, -0.0015376, 0.0016140
5: -0.0038828, -0.0002325, -0.0040472, -0.0002790, -0.0036038, 0.0038147
6: -0.0067728, -0.0050023, -0.0068669, -0.0050389, -0.0017339, 0.0018645
7: -0.0039171, -0.0003870, -0.0038119, -0.0001907, -0.0037263, 0.0034249
8: -0.0073722, -0.0001054, -0.0077266, -0.0002309, -0.0071413, 0.0076212
9: 1.0003257, 1.0022068, 1.0003407, 1.0020653, -0.0017396, 0.0018661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
time: 1.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.76 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015587, upper bound: 0.0015588
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015587, upper bound: 0.0015632
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015632, upper bound: 0.0016989
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015632, upper bound: 0.0017430
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015585, upper bound: 0.0015582
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015585, upper bound: 0.0015626
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015630, upper bound: 0.0016981
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015630, upper bound: 0.0017420
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0016981
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0016981
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015626
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.76
Output dim: 9, lower bound: -0.0015582, upper bound: 0.0015582

## BFS IS instance: IS_A1_B1_A2_B1

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015202, upper bound: 0.0016209
time: 1.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015358, upper bound: 0.0016653
time: 1.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015202, upper bound: 0.0016209
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015358, upper bound: 0.0017190
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049766, 0.0008226, -0.0047391, 0.0005960, -0.0055726, 0.0055617
1: 0.0031411, 0.0072423, 0.0032949, 0.0071035, -0.0039624, 0.0039474
2: 0.0087394, 0.0196612, 0.0091941, 0.0192440, -0.0096141, 0.0095884
3: -0.0068998, -0.0020146, -0.0067071, -0.0021938, -0.0047060, 0.0046925
4: 0.0038299, 0.0054582, 0.0038956, 0.0053966, -0.0015042, 0.0015159
5: -0.0039430, -0.0002896, -0.0037981, -0.0004521, -0.0034910, 0.0035085
6: -0.0068023, -0.0050436, -0.0067318, -0.0051139, -0.0016884, 0.0016883
7: -0.0038014, -0.0003530, -0.0037385, -0.0004570, -0.0033444, 0.0033855
8: -0.0074830, -0.0002510, -0.0072069, -0.0005544, -0.0069286, 0.0069559
9: 1.0003421, 1.0020459, 1.0003587, 1.0017505, -0.0014083, 0.0016872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015200, upper bound: 0.0016200
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015356, upper bound: 0.0016646
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049766, 0.0008226, -0.0051029, 0.0007293, -0.0057059, 0.0059255
1: 0.0031411, 0.0072423, 0.0031957, 0.0073574, -0.0042163, 0.0040466
2: 0.0087394, 0.0196612, 0.0085136, 0.0194918, -0.0096627, 0.0100645
3: -0.0068998, -0.0020146, -0.0068226, -0.0018850, -0.0050149, 0.0048080
4: 0.0038299, 0.0054582, 0.0038572, 0.0054998, -0.0015367, 0.0014852
5: -0.0039430, -0.0002896, -0.0038828, -0.0002325, -0.0037105, 0.0035932
6: -0.0068023, -0.0050436, -0.0067728, -0.0050023, -0.0018000, 0.0017293
7: -0.0038014, -0.0003530, -0.0039171, -0.0003870, -0.0034144, 0.0035641
8: -0.0074830, -0.0002510, -0.0073722, -0.0001054, -0.0073776, 0.0071212
9: 1.0003421, 1.0020459, 1.0003257, 1.0022068, -0.0018647, 0.0017202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015200, upper bound: 0.0016721
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015356, upper bound: 0.0017182
time: 1.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1

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
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016196
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015351, upper bound: 0.0016646
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0051029, 0.0007293, -0.0047391, 0.0005960, -0.0056989, 0.0054684
1: 0.0031957, 0.0073574, 0.0032949, 0.0071035, -0.0039078, 0.0040625
2: 0.0085136, 0.0194918, 0.0091941, 0.0192440, -0.0097720, 0.0093497
3: -0.0068226, -0.0018850, -0.0067071, -0.0021938, -0.0046287, 0.0048221
4: 0.0038572, 0.0054998, 0.0038956, 0.0053966, -0.0014941, 0.0015576
5: -0.0038828, -0.0002325, -0.0037981, -0.0004521, -0.0034307, 0.0035656
6: -0.0067728, -0.0050023, -0.0067318, -0.0051139, -0.0016589, 0.0017295
7: -0.0039171, -0.0003870, -0.0037385, -0.0004570, -0.0034601, 0.0033514
8: -0.0073722, -0.0001054, -0.0072069, -0.0005544, -0.0068178, 0.0071014
9: 1.0003257, 1.0022068, 1.0003587, 1.0017505, -0.0014248, 0.0018481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016195
time: 1.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015351, upper bound: 0.0016646
time: 1.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.49 seconds
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015202, upper bound: 0.0016209
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015358, upper bound: 0.0016653
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015202, upper bound: 0.0016209
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015358, upper bound: 0.0017190
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015200, upper bound: 0.0016200
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015356, upper bound: 0.0016646
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015200, upper bound: 0.0016721
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015356, upper bound: 0.0017182
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016196
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015351, upper bound: 0.0016646
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015195, upper bound: 0.0016195
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 9, lower bound: -0.0015351, upper bound: 0.0016646

## BFS IS instance: IS_A1_B1_A2_B1_A1

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0016131
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0016209
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

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

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015053, upper bound: 0.0016370
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015053, upper bound: 0.0016653
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

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
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016678, upper bound: 0.0016697
time: 1.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016678, upper bound: 0.0016727
time: 1.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

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
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016698, upper bound: 0.0017068
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016698, upper bound: 0.0017190
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0047098, 0.0005948, -0.0052966, 0.0054829
1: 0.0032021, 0.0070349, 0.0032969, 0.0070813, -0.0038792, 0.0037380
2: 0.0092501, 0.0195841, 0.0092481, 0.0192416, -0.0091016, 0.0094186
3: -0.0068406, -0.0022609, -0.0067064, -0.0022202, -0.0046204, 0.0044455
4: 0.0038494, 0.0053766, 0.0038959, 0.0053880, -0.0014827, 0.0014392
5: -0.0039186, -0.0004460, -0.0037967, -0.0004687, -0.0034499, 0.0033506
6: -0.0067835, -0.0051295, -0.0067316, -0.0051229, -0.0016606, 0.0016021
7: -0.0036297, -0.0004172, -0.0037199, -0.0004591, -0.0031706, 0.0033028
8: -0.0074341, -0.0005859, -0.0072052, -0.0005900, -0.0068441, 0.0066193
9: 1.0003712, 1.0016996, 1.0003618, 1.0017141, -0.0013429, 0.0013378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014995, upper bound: 0.0016111
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014995, upper bound: 0.0016199
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0047391, 0.0005960, -0.0054432, 0.0055549
1: 0.0031500, 0.0071585, 0.0032949, 0.0071035, -0.0039535, 0.0038636
2: 0.0089850, 0.0196473, 0.0091941, 0.0192440, -0.0093032, 0.0095729
3: -0.0068959, -0.0021200, -0.0067071, -0.0021938, -0.0047021, 0.0045871
4: 0.0038312, 0.0054226, 0.0038956, 0.0053966, -0.0015023, 0.0014606
5: -0.0039354, -0.0003724, -0.0037981, -0.0004521, -0.0034833, 0.0034257
6: -0.0068006, -0.0050825, -0.0067318, -0.0051139, -0.0016867, 0.0016493
7: -0.0037344, -0.0003632, -0.0037385, -0.0004570, -0.0032774, 0.0033753
8: -0.0074731, -0.0004146, -0.0072069, -0.0005544, -0.0069187, 0.0067923
9: 1.0003535, 1.0018845, 1.0003587, 1.0017505, -0.0013970, 0.0015258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015045, upper bound: 0.0016350
time: 1.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015045, upper bound: 0.0016646
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0050738, 0.0007282, -0.0054299, 0.0058468
1: 0.0032021, 0.0070349, 0.0031976, 0.0073356, -0.0041334, 0.0038373
2: 0.0092501, 0.0195841, 0.0085676, 0.0194894, -0.0091441, 0.0098992
3: -0.0068406, -0.0022609, -0.0068220, -0.0019111, -0.0049295, 0.0045611
4: 0.0038494, 0.0053766, 0.0038574, 0.0054912, -0.0015099, 0.0014056
5: -0.0039186, -0.0004460, -0.0038814, -0.0002495, -0.0036692, 0.0034353
6: -0.0067835, -0.0051295, -0.0067726, -0.0050113, -0.0017722, 0.0016431
7: -0.0036297, -0.0004172, -0.0038987, -0.0003891, -0.0032406, 0.0034815
8: -0.0074341, -0.0005859, -0.0073705, -0.0001407, -0.0072934, 0.0067846
9: 1.0003712, 1.0016996, 1.0003288, 1.0021704, -0.0017992, 0.0013708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016677, upper bound: 0.0016691
time: 1.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016677, upper bound: 0.0016721
time: 1.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0051029, 0.0007293, -0.0055766, 0.0059188
1: 0.0031500, 0.0071585, 0.0031957, 0.0073574, -0.0042074, 0.0039628
2: 0.0089850, 0.0196473, 0.0085136, 0.0194918, -0.0093544, 0.0100492
3: -0.0068959, -0.0021200, -0.0068226, -0.0018850, -0.0050110, 0.0047025
4: 0.0038312, 0.0054226, 0.0038572, 0.0054998, -0.0015348, 0.0014293
5: -0.0039354, -0.0003724, -0.0038828, -0.0002325, -0.0037029, 0.0035104
6: -0.0068006, -0.0050825, -0.0067728, -0.0050023, -0.0017982, 0.0016903
7: -0.0037344, -0.0003632, -0.0039171, -0.0003870, -0.0033474, 0.0035539
8: -0.0074731, -0.0004146, -0.0073722, -0.0001054, -0.0073677, 0.0069576
9: 1.0003535, 1.0018845, 1.0003257, 1.0022068, -0.0018533, 0.0015588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016696, upper bound: 0.0017059
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016696, upper bound: 0.0017182
time: 1.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015001, upper bound: 0.0016126
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015001, upper bound: 0.0016202
time: 1.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015048, upper bound: 0.0016369
time: 1.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015048, upper bound: 0.0016653
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0048378, 0.0006956, -0.0047098, 0.0005948, -0.0054326, 0.0054055
1: 0.0032440, 0.0071585, 0.0032969, 0.0070813, -0.0038373, 0.0038616
2: 0.0090055, 0.0194335, 0.0092481, 0.0192416, -0.0092688, 0.0091997
3: -0.0067841, -0.0021219, -0.0067064, -0.0022202, -0.0045639, 0.0045845
4: 0.0038693, 0.0054216, 0.0038959, 0.0053880, -0.0014707, 0.0014802
5: -0.0038634, -0.0003846, -0.0037967, -0.0004687, -0.0033947, 0.0034121
6: -0.0067606, -0.0050845, -0.0067316, -0.0051229, -0.0016377, 0.0016471
7: -0.0037500, -0.0004389, -0.0037199, -0.0004591, -0.0032909, 0.0032811
8: -0.0073326, -0.0004285, -0.0072052, -0.0005900, -0.0067426, 0.0067767
9: 1.0003529, 1.0018735, 1.0003618, 1.0017141, -0.0013613, 0.0015117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014994, upper bound: 0.0016110
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014994, upper bound: 0.0016195
time: 1.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0047391, 0.0005960, -0.0055643, 0.0054614
1: 0.0032052, 0.0072674, 0.0032949, 0.0071035, -0.0038983, 0.0039725
2: 0.0087682, 0.0194774, 0.0091941, 0.0192440, -0.0094372, 0.0093337
3: -0.0068185, -0.0019956, -0.0067071, -0.0021938, -0.0046247, 0.0047115
4: 0.0038585, 0.0054625, 0.0038956, 0.0053966, -0.0014922, 0.0014851
5: -0.0038752, -0.0003169, -0.0037981, -0.0004521, -0.0034231, 0.0034812
6: -0.0067711, -0.0050429, -0.0067318, -0.0051139, -0.0016572, 0.0016889
7: -0.0038472, -0.0003972, -0.0037385, -0.0004570, -0.0033902, 0.0033412
8: -0.0073620, -0.0002743, -0.0072069, -0.0005544, -0.0068076, 0.0069326
9: 1.0003378, 1.0020391, 1.0003587, 1.0017505, -0.0014126, 0.0016804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015043, upper bound: 0.0016350
time: 1.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015043, upper bound: 0.0016647
time: 1.19 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.88 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0016131
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015004, upper bound: 0.0016209
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015053, upper bound: 0.0016370
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015053, upper bound: 0.0016653
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0016678, upper bound: 0.0016697
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0016678, upper bound: 0.0016727
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0016698, upper bound: 0.0017068
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0016698, upper bound: 0.0017190
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0014995, upper bound: 0.0016111
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0014995, upper bound: 0.0016199
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015045, upper bound: 0.0016350
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015045, upper bound: 0.0016646
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0016677, upper bound: 0.0016691
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0016677, upper bound: 0.0016721
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0016696, upper bound: 0.0017059
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0016696, upper bound: 0.0017182
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015001, upper bound: 0.0016126
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015001, upper bound: 0.0016202
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015048, upper bound: 0.0016369
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015048, upper bound: 0.0016653
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0014994, upper bound: 0.0016110
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0014994, upper bound: 0.0016195
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015043, upper bound: 0.0016350
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 9, lower bound: -0.0015043, upper bound: 0.0016647

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0043260, 0.0006410, -0.0053427, 0.0050991
1: 0.0032021, 0.0070349, 0.0033055, 0.0067766, -0.0035745, 0.0037294
2: 0.0092501, 0.0195841, 0.0099497, 0.0193339, -0.0091329, 0.0086835
3: -0.0068406, -0.0022609, -0.0067276, -0.0025779, -0.0042627, 0.0044667
4: 0.0038494, 0.0053766, 0.0038874, 0.0052702, -0.0013513, 0.0014120
5: -0.0039186, -0.0004460, -0.0038326, -0.0006751, -0.0032436, 0.0033866
6: -0.0067835, -0.0051295, -0.0067435, -0.0052440, -0.0015395, 0.0016141
7: -0.0036297, -0.0004172, -0.0034426, -0.0004929, -0.0031368, 0.0030254
8: -0.0074341, -0.0005859, -0.0072679, -0.0010480, -0.0063861, 0.0066820
9: 1.0003712, 1.0016996, 1.0004048, 1.0012336, -0.0008624, 0.0012947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014427, upper bound: 0.0015675
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014379, upper bound: 0.0015335
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0044610, 0.0006786, -0.0053804, 0.0052341
1: 0.0032021, 0.0070349, 0.0032577, 0.0068947, -0.0036926, 0.0037773
2: 0.0092501, 0.0195841, 0.0097068, 0.0193948, -0.0092181, 0.0089649
3: -0.0068406, -0.0022609, -0.0067739, -0.0024416, -0.0043990, 0.0045130
4: 0.0038494, 0.0053766, 0.0038723, 0.0053146, -0.0013967, 0.0014297
5: -0.0039186, -0.0004460, -0.0038447, -0.0006047, -0.0033139, 0.0033987
6: -0.0067835, -0.0051295, -0.0067572, -0.0052002, -0.0015834, 0.0016277
7: -0.0036297, -0.0004172, -0.0035488, -0.0004399, -0.0031898, 0.0031316
8: -0.0074341, -0.0005859, -0.0073055, -0.0008898, -0.0065443, 0.0067196
9: 1.0003712, 1.0016996, 1.0003880, 1.0014049, -0.0010337, 0.0013115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014427, upper bound: 0.0015694
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014379, upper bound: 0.0015370
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0043260, 0.0006410, -0.0054882, 0.0051419
1: 0.0031500, 0.0071585, 0.0033055, 0.0067766, -0.0036266, 0.0038530
2: 0.0089850, 0.0196473, 0.0099497, 0.0193339, -0.0094114, 0.0087822
3: -0.0068959, -0.0021200, -0.0067276, -0.0025779, -0.0043180, 0.0046076
4: 0.0038312, 0.0054226, 0.0038874, 0.0052702, -0.0013625, 0.0014516
5: -0.0039354, -0.0003724, -0.0038326, -0.0006751, -0.0032603, 0.0034603
6: -0.0068006, -0.0050825, -0.0067435, -0.0052440, -0.0015566, 0.0016610
7: -0.0037344, -0.0003632, -0.0034426, -0.0004929, -0.0032415, 0.0030794
8: -0.0074731, -0.0004146, -0.0072679, -0.0010480, -0.0064251, 0.0068533
9: 1.0003535, 1.0018845, 1.0004048, 1.0012336, -0.0008801, 0.0014796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014483, upper bound: 0.0015871
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014422, upper bound: 0.0015441
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0044610, 0.0006786, -0.0055259, 0.0052769
1: 0.0031500, 0.0071585, 0.0032577, 0.0068947, -0.0037447, 0.0039008
2: 0.0089850, 0.0196473, 0.0097068, 0.0193948, -0.0094059, 0.0089691
3: -0.0068959, -0.0021200, -0.0067739, -0.0024416, -0.0044543, 0.0046539
4: 0.0038312, 0.0054226, 0.0038723, 0.0053146, -0.0013826, 0.0014429
5: -0.0039354, -0.0003724, -0.0038447, -0.0006047, -0.0033307, 0.0034723
6: -0.0068006, -0.0050825, -0.0067572, -0.0052002, -0.0016004, 0.0016747
7: -0.0037344, -0.0003632, -0.0035488, -0.0004399, -0.0032945, 0.0031856
8: -0.0074731, -0.0004146, -0.0073055, -0.0008898, -0.0065833, 0.0068909
9: 1.0003535, 1.0018845, 1.0003880, 1.0014049, -0.0010514, 0.0014964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014483, upper bound: 0.0016000
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014422, upper bound: 0.0015551
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0047018, 0.0007730, -0.0054748, 0.0054748
1: 0.0032021, 0.0070349, 0.0032021, 0.0070349, -0.0038328, 0.0038328
2: 0.0092501, 0.0195841, 0.0092501, 0.0195841, -0.0091805, 0.0091805
3: -0.0068406, -0.0022609, -0.0068406, -0.0022609, -0.0045797, 0.0045797
4: 0.0038494, 0.0053766, 0.0038494, 0.0053766, -0.0013798, 0.0013798
5: -0.0039186, -0.0004460, -0.0039186, -0.0004460, -0.0034726, 0.0034726
6: -0.0067835, -0.0051295, -0.0067835, -0.0051295, -0.0016541, 0.0016541
7: -0.0036297, -0.0004172, -0.0036297, -0.0004172, -0.0032125, 0.0032125
8: -0.0074341, -0.0005859, -0.0074341, -0.0005859, -0.0068482, 0.0068482
9: 1.0003712, 1.0016996, 1.0003712, 1.0016996, -0.0013283, 0.0013283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016555
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014111, upper bound: 0.0016098
time: 1.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0048472, 0.0008159, -0.0055176, 0.0056202
1: 0.0032021, 0.0070349, 0.0031500, 0.0071585, -0.0039564, 0.0038849
2: 0.0092501, 0.0195841, 0.0089850, 0.0196473, -0.0092757, 0.0094651
3: -0.0068406, -0.0022609, -0.0068959, -0.0021200, -0.0047206, 0.0046350
4: 0.0038494, 0.0053766, 0.0038312, 0.0054226, -0.0014243, 0.0013963
5: -0.0039186, -0.0004460, -0.0039354, -0.0003724, -0.0035463, 0.0034893
6: -0.0067835, -0.0051295, -0.0068006, -0.0050825, -0.0017010, 0.0016711
7: -0.0036297, -0.0004172, -0.0037344, -0.0003632, -0.0032665, 0.0033172
8: -0.0074341, -0.0005859, -0.0074731, -0.0004146, -0.0070195, 0.0068872
9: 1.0003712, 1.0016996, 1.0003535, 1.0018845, -0.0015132, 0.0013461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016570
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014111, upper bound: 0.0016122
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0047018, 0.0007730, -0.0056202, 0.0055176
1: 0.0031500, 0.0071585, 0.0032021, 0.0070349, -0.0038849, 0.0039564
2: 0.0089850, 0.0196473, 0.0092501, 0.0195841, -0.0094651, 0.0092757
3: -0.0068959, -0.0021200, -0.0068406, -0.0022609, -0.0046350, 0.0047206
4: 0.0038312, 0.0054226, 0.0038494, 0.0053766, -0.0013963, 0.0014243
5: -0.0039354, -0.0003724, -0.0039186, -0.0004460, -0.0034893, 0.0035463
6: -0.0068006, -0.0050825, -0.0067835, -0.0051295, -0.0016711, 0.0017010
7: -0.0037344, -0.0003632, -0.0036297, -0.0004172, -0.0033172, 0.0032665
8: -0.0074731, -0.0004146, -0.0074341, -0.0005859, -0.0068872, 0.0070195
9: 1.0003535, 1.0018845, 1.0003712, 1.0016996, -0.0013461, 0.0015132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016869
time: 1.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016228
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0048472, 0.0008159, -0.0056631, 0.0056631
1: 0.0031500, 0.0071585, 0.0031500, 0.0071585, -0.0040085, 0.0040085
2: 0.0089850, 0.0196473, 0.0089850, 0.0196473, -0.0094692, 0.0094692
3: -0.0068959, -0.0021200, -0.0068959, -0.0021200, -0.0047759, 0.0047759
4: 0.0038312, 0.0054226, 0.0038312, 0.0054226, -0.0014125, 0.0014125
5: -0.0039354, -0.0003724, -0.0039354, -0.0003724, -0.0035630, 0.0035630
6: -0.0068006, -0.0050825, -0.0068006, -0.0050825, -0.0017180, 0.0017180
7: -0.0037344, -0.0003632, -0.0037344, -0.0003632, -0.0033712, 0.0033712
8: -0.0074731, -0.0004146, -0.0074731, -0.0004146, -0.0070585, 0.0070585
9: 1.0003535, 1.0018845, 1.0003535, 1.0018845, -0.0015310, 0.0015310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016969
time: 1.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016284
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0044732, 0.0005648, -0.0052666, 0.0052463
1: 0.0032021, 0.0070349, 0.0033426, 0.0069070, -0.0037049, 0.0036924
2: 0.0092501, 0.0195841, 0.0096881, 0.0191906, -0.0090363, 0.0089766
3: -0.0068406, -0.0022609, -0.0066695, -0.0024291, -0.0044114, 0.0044086
4: 0.0038494, 0.0053766, 0.0039071, 0.0053186, -0.0014166, 0.0014287
5: -0.0039186, -0.0004460, -0.0037820, -0.0006056, -0.0033130, 0.0033360
6: -0.0067835, -0.0051295, -0.0067209, -0.0051960, -0.0015875, 0.0015914
7: -0.0036297, -0.0004172, -0.0035731, -0.0005080, -0.0031216, 0.0031560
8: -0.0074341, -0.0005859, -0.0071722, -0.0008796, -0.0065545, 0.0065863
9: 1.0003712, 1.0016996, 1.0003859, 1.0014188, -0.0010476, 0.0013137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014407, upper bound: 0.0015647
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014359, upper bound: 0.0015322
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0046036, 0.0005888, -0.0052905, 0.0053766
1: 0.0032021, 0.0070349, 0.0033049, 0.0070156, -0.0038135, 0.0037300
2: 0.0092501, 0.0195841, 0.0094494, 0.0192293, -0.0090881, 0.0092277
3: -0.0068406, -0.0022609, -0.0067024, -0.0023016, -0.0045390, 0.0044415
4: 0.0038494, 0.0053766, 0.0038970, 0.0053602, -0.0014546, 0.0014376
5: -0.0039186, -0.0004460, -0.0037904, -0.0005382, -0.0033804, 0.0033444
6: -0.0067835, -0.0051295, -0.0067300, -0.0051541, -0.0016294, 0.0016005
7: -0.0036297, -0.0004172, -0.0036721, -0.0004671, -0.0031626, 0.0032550
8: -0.0074341, -0.0005859, -0.0071967, -0.0007234, -0.0067107, 0.0066109
9: 1.0003712, 1.0016996, 1.0003704, 1.0015838, -0.0012126, 0.0013292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014407, upper bound: 0.0015663
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014359, upper bound: 0.0015359
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0044732, 0.0005648, -0.0054121, 0.0052891
1: 0.0031500, 0.0071585, 0.0033426, 0.0069070, -0.0037570, 0.0038159
2: 0.0089850, 0.0196473, 0.0096881, 0.0191906, -0.0093148, 0.0090753
3: -0.0068959, -0.0021200, -0.0066695, -0.0024291, -0.0044668, 0.0045495
4: 0.0038312, 0.0054226, 0.0039071, 0.0053186, -0.0014278, 0.0014683
5: -0.0039354, -0.0003724, -0.0037820, -0.0006056, -0.0033298, 0.0034097
6: -0.0068006, -0.0050825, -0.0067209, -0.0051960, -0.0016045, 0.0016383
7: -0.0037344, -0.0003632, -0.0035731, -0.0005080, -0.0032264, 0.0032100
8: -0.0074731, -0.0004146, -0.0071722, -0.0008796, -0.0065935, 0.0067576
9: 1.0003535, 1.0018845, 1.0003859, 1.0014188, -0.0010654, 0.0014986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014464, upper bound: 0.0015826
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014401, upper bound: 0.0015421
time: 1.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0046036, 0.0005888, -0.0054360, 0.0054195
1: 0.0031500, 0.0071585, 0.0033049, 0.0070156, -0.0038656, 0.0038536
2: 0.0089850, 0.0196473, 0.0094494, 0.0192293, -0.0092866, 0.0092433
3: -0.0068959, -0.0021200, -0.0067024, -0.0023016, -0.0045944, 0.0045824
4: 0.0038312, 0.0054226, 0.0038970, 0.0053602, -0.0014465, 0.0014587
5: -0.0039354, -0.0003724, -0.0037904, -0.0005382, -0.0033972, 0.0034180
6: -0.0068006, -0.0050825, -0.0067300, -0.0051541, -0.0016465, 0.0016475
7: -0.0037344, -0.0003632, -0.0036721, -0.0004671, -0.0032673, 0.0033090
8: -0.0074731, -0.0004146, -0.0071967, -0.0007234, -0.0067498, 0.0067822
9: 1.0003535, 1.0018845, 1.0003704, 1.0015838, -0.0012304, 0.0015141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014464, upper bound: 0.0015973
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014401, upper bound: 0.0015543
time: 1.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0048378, 0.0006956, -0.0053974, 0.0056108
1: 0.0032021, 0.0070349, 0.0032440, 0.0071585, -0.0039563, 0.0037909
2: 0.0092501, 0.0195841, 0.0090055, 0.0194335, -0.0090800, 0.0094612
3: -0.0068406, -0.0022609, -0.0067841, -0.0021219, -0.0047187, 0.0045232
4: 0.0038494, 0.0053766, 0.0038693, 0.0054216, -0.0014444, 0.0013958
5: -0.0039186, -0.0004460, -0.0038634, -0.0003846, -0.0035340, 0.0034174
6: -0.0067835, -0.0051295, -0.0067606, -0.0050845, -0.0016990, 0.0016311
7: -0.0036297, -0.0004172, -0.0037500, -0.0004389, -0.0031908, 0.0033328
8: -0.0074341, -0.0005859, -0.0073326, -0.0004285, -0.0070056, 0.0067467
9: 1.0003712, 1.0016996, 1.0003529, 1.0018735, -0.0015023, 0.0013467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014218, upper bound: 0.0016547
time: 1.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014087, upper bound: 0.0016092
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0047018, 0.0007730, -0.0049683, 0.0007223, -0.0054241, 0.0057413
1: 0.0032021, 0.0070349, 0.0032052, 0.0072674, -0.0040653, 0.0038297
2: 0.0092501, 0.0195841, 0.0087682, 0.0194774, -0.0091308, 0.0097039
3: -0.0068406, -0.0022609, -0.0068185, -0.0019956, -0.0048450, 0.0045576
4: 0.0038494, 0.0053766, 0.0038585, 0.0054625, -0.0014810, 0.0014041
5: -0.0039186, -0.0004460, -0.0038752, -0.0003169, -0.0036017, 0.0034292
6: -0.0067835, -0.0051295, -0.0067711, -0.0050429, -0.0017406, 0.0016416
7: -0.0036297, -0.0004172, -0.0038472, -0.0003972, -0.0032325, 0.0034300
8: -0.0074341, -0.0005859, -0.0073620, -0.0002743, -0.0071598, 0.0067761
9: 1.0003712, 1.0016996, 1.0003378, 1.0020391, -0.0016679, 0.0013617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014218, upper bound: 0.0016562
time: 1.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014087, upper bound: 0.0016114
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0048378, 0.0006956, -0.0055429, 0.0056536
1: 0.0031500, 0.0071585, 0.0032440, 0.0071585, -0.0040085, 0.0039145
2: 0.0089850, 0.0196473, 0.0090055, 0.0194335, -0.0093646, 0.0095564
3: -0.0068959, -0.0021200, -0.0067841, -0.0021219, -0.0047740, 0.0046641
4: 0.0038312, 0.0054226, 0.0038693, 0.0054216, -0.0014610, 0.0014403
5: -0.0039354, -0.0003724, -0.0038634, -0.0003846, -0.0035508, 0.0034910
6: -0.0068006, -0.0050825, -0.0067606, -0.0050845, -0.0017161, 0.0016781
7: -0.0037344, -0.0003632, -0.0037500, -0.0004389, -0.0032956, 0.0033868
8: -0.0074731, -0.0004146, -0.0073326, -0.0004285, -0.0070446, 0.0069180
9: 1.0003535, 1.0018845, 1.0003529, 1.0018735, -0.0015200, 0.0015316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016137, upper bound: 0.0016852
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016109, upper bound: 0.0016224
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0048472, 0.0008159, -0.0049683, 0.0007223, -0.0055695, 0.0057841
1: 0.0031500, 0.0071585, 0.0032052, 0.0072674, -0.0041174, 0.0039533
2: 0.0089850, 0.0196473, 0.0087682, 0.0194774, -0.0093382, 0.0097202
3: -0.0068959, -0.0021200, -0.0068185, -0.0019956, -0.0049004, 0.0046985
4: 0.0038312, 0.0054226, 0.0038585, 0.0054625, -0.0014756, 0.0014275
5: -0.0039354, -0.0003724, -0.0038752, -0.0003169, -0.0036185, 0.0035028
6: -0.0068006, -0.0050825, -0.0067711, -0.0050429, -0.0017577, 0.0016885
7: -0.0037344, -0.0003632, -0.0038472, -0.0003972, -0.0033372, 0.0034840
8: -0.0074731, -0.0004146, -0.0073620, -0.0002743, -0.0071988, 0.0069474
9: 1.0003535, 1.0018845, 1.0003378, 1.0020391, -0.0016856, 0.0015466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016137, upper bound: 0.0016957
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016109, upper bound: 0.0016278
time: 1.95 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0048378, 0.0006956, -0.0043260, 0.0006410, -0.0054787, 0.0050217
1: 0.0032440, 0.0071585, 0.0033055, 0.0067766, -0.0035326, 0.0038529
2: 0.0090055, 0.0194335, 0.0099497, 0.0193339, -0.0094043, 0.0085780
3: -0.0067841, -0.0021219, -0.0067276, -0.0025779, -0.0042062, 0.0046057
4: 0.0038693, 0.0054216, 0.0038874, 0.0052702, -0.0013550, 0.0014700
5: -0.0038634, -0.0003846, -0.0038326, -0.0006751, -0.0031883, 0.0034480
6: -0.0067606, -0.0050845, -0.0067435, -0.0052440, -0.0015166, 0.0016590
7: -0.0037500, -0.0004389, -0.0034426, -0.0004929, -0.0032571, 0.0030037
8: -0.0073326, -0.0004285, -0.0072679, -0.0010480, -0.0062845, 0.0068394
9: 1.0003529, 1.0018735, 1.0004048, 1.0012336, -0.0008807, 0.0014687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014424, upper bound: 0.0015673
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014373, upper bound: 0.0015330
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048378, 0.0006956, -0.0044610, 0.0006786, -0.0055164, 0.0051567
1: 0.0032440, 0.0071585, 0.0032577, 0.0068947, -0.0036507, 0.0039008
2: 0.0090055, 0.0194335, 0.0097068, 0.0193948, -0.0094895, 0.0088594
3: -0.0067841, -0.0021219, -0.0067739, -0.0024416, -0.0043425, 0.0046520
4: 0.0038693, 0.0054216, 0.0038723, 0.0053146, -0.0014003, 0.0014877
5: -0.0038634, -0.0003846, -0.0038447, -0.0006047, -0.0032587, 0.0034601
6: -0.0067606, -0.0050845, -0.0067572, -0.0052002, -0.0015605, 0.0016727
7: -0.0037500, -0.0004389, -0.0035488, -0.0004399, -0.0033100, 0.0031100
8: -0.0073326, -0.0004285, -0.0073055, -0.0008898, -0.0064427, 0.0068770
9: 1.0003529, 1.0018735, 1.0003880, 1.0014049, -0.0010520, 0.0014855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014424, upper bound: 0.0015692
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014373, upper bound: 0.0015367
time: 2.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0043260, 0.0006410, -0.0056092, 0.0050483
1: 0.0032052, 0.0072674, 0.0033055, 0.0067766, -0.0035714, 0.0039619
2: 0.0087682, 0.0194774, 0.0099497, 0.0193339, -0.0096478, 0.0086294
3: -0.0068185, -0.0019956, -0.0067276, -0.0025779, -0.0042406, 0.0047321
4: 0.0038585, 0.0054625, 0.0038874, 0.0052702, -0.0013591, 0.0015021
5: -0.0038752, -0.0003169, -0.0038326, -0.0006751, -0.0032001, 0.0035158
6: -0.0067711, -0.0050429, -0.0067435, -0.0052440, -0.0015270, 0.0017006
7: -0.0038472, -0.0003972, -0.0034426, -0.0004929, -0.0033543, 0.0030454
8: -0.0073620, -0.0002743, -0.0072679, -0.0010480, -0.0063140, 0.0069936
9: 1.0003378, 1.0020391, 1.0004048, 1.0012336, -0.0008957, 0.0016342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014477, upper bound: 0.0015871
time: 2.07 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014414, upper bound: 0.0015440
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0044610, 0.0006786, -0.0056469, 0.0051834
1: 0.0032052, 0.0072674, 0.0032577, 0.0068947, -0.0036895, 0.0040097
2: 0.0087682, 0.0194774, 0.0097068, 0.0193948, -0.0096546, 0.0088348
3: -0.0068185, -0.0019956, -0.0067739, -0.0024416, -0.0043769, 0.0047784
4: 0.0038585, 0.0054625, 0.0038723, 0.0053146, -0.0013854, 0.0014998
5: -0.0038752, -0.0003169, -0.0038447, -0.0006047, -0.0032705, 0.0035278
6: -0.0067711, -0.0050429, -0.0067572, -0.0052002, -0.0015709, 0.0017143
7: -0.0038472, -0.0003972, -0.0035488, -0.0004399, -0.0034073, 0.0031516
8: -0.0073620, -0.0002743, -0.0073055, -0.0008898, -0.0064722, 0.0070312
9: 1.0003378, 1.0020391, 1.0003880, 1.0014049, -0.0010670, 0.0016510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014477, upper bound: 0.0016000
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014414, upper bound: 0.0015550
time: 1.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0048378, 0.0006956, -0.0044732, 0.0005648, -0.0054026, 0.0051689
1: 0.0032440, 0.0071585, 0.0033426, 0.0069070, -0.0036630, 0.0038159
2: 0.0090055, 0.0194335, 0.0096881, 0.0191906, -0.0091844, 0.0087565
3: -0.0067841, -0.0021219, -0.0066695, -0.0024291, -0.0043550, 0.0045476
4: 0.0038693, 0.0054216, 0.0039071, 0.0053186, -0.0013994, 0.0014590
5: -0.0038634, -0.0003846, -0.0037820, -0.0006056, -0.0032578, 0.0033974
6: -0.0067606, -0.0050845, -0.0067209, -0.0051960, -0.0015646, 0.0016364
7: -0.0037500, -0.0004389, -0.0035731, -0.0005080, -0.0032419, 0.0031343
8: -0.0073326, -0.0004285, -0.0071722, -0.0008796, -0.0064530, 0.0067437
9: 1.0003529, 1.0018735, 1.0003859, 1.0014188, -0.0010660, 0.0014876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014406, upper bound: 0.0015647
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014357, upper bound: 0.0015322
time: 1.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048378, 0.0006956, -0.0046036, 0.0005888, -0.0054265, 0.0052992
1: 0.0032440, 0.0071585, 0.0033049, 0.0070156, -0.0037716, 0.0038536
2: 0.0090055, 0.0194335, 0.0094494, 0.0192293, -0.0092550, 0.0090175
3: -0.0067841, -0.0021219, -0.0067024, -0.0023016, -0.0044826, 0.0045805
4: 0.0038693, 0.0054216, 0.0038970, 0.0053602, -0.0014451, 0.0014786
5: -0.0038634, -0.0003846, -0.0037904, -0.0005382, -0.0033252, 0.0034058
6: -0.0067606, -0.0050845, -0.0067300, -0.0051541, -0.0016065, 0.0016455
7: -0.0037500, -0.0004389, -0.0036721, -0.0004671, -0.0032829, 0.0032333
8: -0.0073326, -0.0004285, -0.0071967, -0.0007234, -0.0066092, 0.0067683
9: 1.0003529, 1.0018735, 1.0003704, 1.0015838, -0.0012310, 0.0015031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014406, upper bound: 0.0015663
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014357, upper bound: 0.0015358
time: 1.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0044732, 0.0005648, -0.0055331, 0.0051956
1: 0.0032052, 0.0072674, 0.0033426, 0.0069070, -0.0037018, 0.0039248
2: 0.0087682, 0.0194774, 0.0096881, 0.0191906, -0.0094496, 0.0088353
3: -0.0068185, -0.0019956, -0.0066695, -0.0024291, -0.0043894, 0.0046740
4: 0.0038585, 0.0054625, 0.0039071, 0.0053186, -0.0014118, 0.0014992
5: -0.0038752, -0.0003169, -0.0037820, -0.0006056, -0.0032696, 0.0034651
6: -0.0067711, -0.0050429, -0.0067209, -0.0051960, -0.0015750, 0.0016780
7: -0.0038472, -0.0003972, -0.0035731, -0.0005080, -0.0033392, 0.0031759
8: -0.0073620, -0.0002743, -0.0071722, -0.0008796, -0.0064825, 0.0068979
9: 1.0003378, 1.0020391, 1.0003859, 1.0014188, -0.0010810, 0.0016532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014461, upper bound: 0.0015826
time: 1.49 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014399, upper bound: 0.0015421
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049683, 0.0007223, -0.0046036, 0.0005888, -0.0055570, 0.0053259
1: 0.0032052, 0.0072674, 0.0033049, 0.0070156, -0.0038104, 0.0039625
2: 0.0087682, 0.0194774, 0.0094494, 0.0192293, -0.0094207, 0.0089996
3: -0.0068185, -0.0019956, -0.0067024, -0.0023016, -0.0045169, 0.0047069
4: 0.0038585, 0.0054625, 0.0038970, 0.0053602, -0.0014241, 0.0014831
5: -0.0038752, -0.0003169, -0.0037904, -0.0005382, -0.0033370, 0.0034735
6: -0.0067711, -0.0050429, -0.0067300, -0.0051541, -0.0016169, 0.0016871
7: -0.0038472, -0.0003972, -0.0036721, -0.0004671, -0.0033801, 0.0032749
8: -0.0073620, -0.0002743, -0.0071967, -0.0007234, -0.0066387, 0.0069224
9: 1.0003378, 1.0020391, 1.0003704, 1.0015838, -0.0012460, 0.0016687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014461, upper bound: 0.0015973
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014399, upper bound: 0.0015543
time: 1.40 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.81 seconds
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014427, upper bound: 0.0015675
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014379, upper bound: 0.0015335
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014427, upper bound: 0.0015694
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014379, upper bound: 0.0015370
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014483, upper bound: 0.0015871
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014422, upper bound: 0.0015441
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014483, upper bound: 0.0016000
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014422, upper bound: 0.0015551
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016555
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014111, upper bound: 0.0016098
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016117, upper bound: 0.0016570
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014111, upper bound: 0.0016122
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016869
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016112, upper bound: 0.0016228
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016141, upper bound: 0.0016969
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016284
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014407, upper bound: 0.0015647
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014359, upper bound: 0.0015322
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014407, upper bound: 0.0015663
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014359, upper bound: 0.0015359
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014464, upper bound: 0.0015826
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014401, upper bound: 0.0015421
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014464, upper bound: 0.0015973
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014401, upper bound: 0.0015543
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014218, upper bound: 0.0016547
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014087, upper bound: 0.0016092
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014218, upper bound: 0.0016562
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014087, upper bound: 0.0016114
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016137, upper bound: 0.0016852
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016109, upper bound: 0.0016224
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016137, upper bound: 0.0016957
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0016109, upper bound: 0.0016278
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014424, upper bound: 0.0015673
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014373, upper bound: 0.0015330
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014424, upper bound: 0.0015692
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014373, upper bound: 0.0015367
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014477, upper bound: 0.0015871
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014414, upper bound: 0.0015440
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014477, upper bound: 0.0016000
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014414, upper bound: 0.0015550
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014406, upper bound: 0.0015647
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014357, upper bound: 0.0015322
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014406, upper bound: 0.0015663
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014357, upper bound: 0.0015358
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014461, upper bound: 0.0015826
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014399, upper bound: 0.0015421
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014461, upper bound: 0.0015973
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 9, lower bound: -0.0014399, upper bound: 0.0015543

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0043260, 0.0006410, -0.0054780, 0.0049741
1: 0.0033019, 0.0071539, 0.0033055, 0.0067766, -0.0034748, 0.0038484
2: 0.0090055, 0.0193522, 0.0099497, 0.0193339, -0.0093913, 0.0084853
3: -0.0067221, -0.0021260, -0.0067276, -0.0025779, -0.0041442, 0.0046016
4: 0.0038875, 0.0054204, 0.0038874, 0.0052702, -0.0013049, 0.0014493
5: -0.0038581, -0.0003803, -0.0038326, -0.0006751, -0.0031831, 0.0034524
6: -0.0067448, -0.0050854, -0.0067435, -0.0052440, -0.0015008, 0.0016581
7: -0.0037320, -0.0005139, -0.0034426, -0.0004929, -0.0032391, 0.0029287
8: -0.0072838, -0.0004284, -0.0072679, -0.0010480, -0.0062358, 0.0068395
9: 1.0003542, 1.0018717, 1.0004048, 1.0012336, -0.0008794, 0.0014669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014303, upper bound: 0.0015692
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014243, upper bound: 0.0015654
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0044610, 0.0006786, -0.0055156, 0.0051091
1: 0.0033019, 0.0071539, 0.0032577, 0.0068947, -0.0035929, 0.0038963
2: 0.0090055, 0.0193522, 0.0097068, 0.0193948, -0.0093857, 0.0086732
3: -0.0067221, -0.0021260, -0.0067739, -0.0024416, -0.0042805, 0.0046479
4: 0.0038875, 0.0054204, 0.0038723, 0.0053146, -0.0013276, 0.0014405
5: -0.0038581, -0.0003803, -0.0038447, -0.0006047, -0.0032534, 0.0034644
6: -0.0067448, -0.0050854, -0.0067572, -0.0052002, -0.0015447, 0.0016718
7: -0.0037320, -0.0005139, -0.0035488, -0.0004399, -0.0032921, 0.0030350
8: -0.0072838, -0.0004284, -0.0073055, -0.0008898, -0.0063940, 0.0068771
9: 1.0003542, 1.0018717, 1.0003880, 1.0014049, -0.0010507, 0.0014837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014551, upper bound: 0.0015550
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014551, upper bound: 0.0015550
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0047018, 0.0007730, -0.0054650, 0.0053109
1: 0.0033514, 0.0070309, 0.0032021, 0.0070349, -0.0036836, 0.0038288
2: 0.0092692, 0.0192927, 0.0092501, 0.0195841, -0.0091615, 0.0088920
3: -0.0066708, -0.0022665, -0.0068406, -0.0022609, -0.0044099, 0.0045741
4: 0.0039042, 0.0053746, 0.0038494, 0.0053766, -0.0013245, 0.0013776
5: -0.0038431, -0.0004534, -0.0039186, -0.0004460, -0.0033971, 0.0034653
6: -0.0067294, -0.0051322, -0.0067835, -0.0051295, -0.0015999, 0.0016514
7: -0.0036275, -0.0005668, -0.0036297, -0.0004172, -0.0032103, 0.0030629
8: -0.0072464, -0.0005987, -0.0074341, -0.0005859, -0.0066605, 0.0068354
9: 1.0003716, 1.0016876, 1.0003712, 1.0016996, -0.0013280, 0.0013164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
time: 1.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0048472, 0.0008159, -0.0055079, 0.0054564
1: 0.0033514, 0.0070309, 0.0031500, 0.0071585, -0.0038071, 0.0038809
2: 0.0092692, 0.0192927, 0.0089850, 0.0196473, -0.0092567, 0.0091766
3: -0.0066708, -0.0022665, -0.0068959, -0.0021200, -0.0045508, 0.0046295
4: 0.0039042, 0.0053746, 0.0038312, 0.0054226, -0.0013691, 0.0013942
5: -0.0038431, -0.0004534, -0.0039354, -0.0003724, -0.0034708, 0.0034820
6: -0.0067294, -0.0051322, -0.0068006, -0.0050825, -0.0016468, 0.0016684
7: -0.0036275, -0.0005668, -0.0037344, -0.0003632, -0.0032643, 0.0031677
8: -0.0072464, -0.0005987, -0.0074731, -0.0004146, -0.0068318, 0.0068744
9: 1.0003716, 1.0016876, 1.0003535, 1.0018845, -0.0015129, 0.0013342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
time: 2.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
time: 1.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016228
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016228
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016228
time: 1.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016228
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016284
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016284
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016284
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016284
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0046036, 0.0005888, -0.0054257, 0.0052517
1: 0.0033019, 0.0071539, 0.0033049, 0.0070156, -0.0037138, 0.0038491
2: 0.0090055, 0.0193522, 0.0094494, 0.0192293, -0.0092664, 0.0089474
3: -0.0067221, -0.0021260, -0.0067024, -0.0023016, -0.0044206, 0.0045764
4: 0.0038875, 0.0054204, 0.0038970, 0.0053602, -0.0013915, 0.0014563
5: -0.0038581, -0.0003803, -0.0037904, -0.0005382, -0.0033199, 0.0034101
6: -0.0067448, -0.0050854, -0.0067300, -0.0051541, -0.0015907, 0.0016446
7: -0.0037320, -0.0005139, -0.0036721, -0.0004671, -0.0032649, 0.0031583
8: -0.0072838, -0.0004284, -0.0071967, -0.0007234, -0.0065605, 0.0067684
9: 1.0003542, 1.0018717, 1.0003704, 1.0015838, -0.0012296, 0.0015013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014545, upper bound: 0.0015543
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014545, upper bound: 0.0015543
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0048378, 0.0006956, -0.0053877, 0.0054469
1: 0.0033514, 0.0070309, 0.0032440, 0.0071585, -0.0038071, 0.0037868
2: 0.0092692, 0.0192927, 0.0090055, 0.0194335, -0.0090610, 0.0091727
3: -0.0066708, -0.0022665, -0.0067841, -0.0021219, -0.0045489, 0.0045177
4: 0.0039042, 0.0053746, 0.0038693, 0.0054216, -0.0013891, 0.0013936
5: -0.0038431, -0.0004534, -0.0038634, -0.0003846, -0.0034585, 0.0034100
6: -0.0067294, -0.0051322, -0.0067606, -0.0050845, -0.0016448, 0.0016285
7: -0.0036275, -0.0005668, -0.0037500, -0.0004389, -0.0031887, 0.0031832
8: -0.0072464, -0.0005987, -0.0073326, -0.0004285, -0.0068179, 0.0067339
9: 1.0003716, 1.0016876, 1.0003529, 1.0018735, -0.0015019, 0.0013348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016089, upper bound: 0.0016092
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016089, upper bound: 0.0016092
time: 1.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0048353, 0.0006653, -0.0056766, 0.0054228
1: 0.0033506, 0.0073251, 0.0032701, 0.0071575, -0.0038068, 0.0040550
2: 0.0087046, 0.0192520, 0.0090103, 0.0193793, -0.0095926, 0.0091734
3: -0.0066545, -0.0019347, -0.0067545, -0.0021233, -0.0045312, 0.0048198
4: 0.0039107, 0.0054817, 0.0038790, 0.0054212, -0.0014209, 0.0015008
5: -0.0038354, -0.0003103, -0.0038486, -0.0003865, -0.0034489, 0.0035383
6: -0.0067229, -0.0050264, -0.0067507, -0.0050852, -0.0016377, 0.0017243
7: -0.0039236, -0.0005583, -0.0037494, -0.0004657, -0.0034579, 0.0031911
8: -0.0072211, -0.0002366, -0.0072976, -0.0004317, -0.0067894, 0.0070610
9: 1.0003281, 1.0020975, 1.0003531, 1.0018706, -0.0015426, 0.0017444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014087, upper bound: 0.0016092
time: 1.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014087, upper bound: 0.0016092
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0049683, 0.0007223, -0.0054143, 0.0055774
1: 0.0033514, 0.0070309, 0.0032052, 0.0072674, -0.0039160, 0.0038257
2: 0.0092692, 0.0192927, 0.0087682, 0.0194774, -0.0091118, 0.0094154
3: -0.0066708, -0.0022665, -0.0068185, -0.0019956, -0.0046753, 0.0045520
4: 0.0039042, 0.0053746, 0.0038585, 0.0054625, -0.0014257, 0.0014019
5: -0.0038431, -0.0004534, -0.0038752, -0.0003169, -0.0035263, 0.0034218
6: -0.0067294, -0.0051322, -0.0067711, -0.0050429, -0.0016865, 0.0016389
7: -0.0036275, -0.0005668, -0.0038472, -0.0003972, -0.0032303, 0.0032804
8: -0.0072464, -0.0005987, -0.0073620, -0.0002743, -0.0069721, 0.0067633
9: 1.0003716, 1.0016876, 1.0003378, 1.0020391, -0.0016675, 0.0013498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016114
time: 1.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016114
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0049659, 0.0006930, -0.0057043, 0.0055533
1: 0.0033506, 0.0073251, 0.0032308, 0.0072664, -0.0039157, 0.0040943
2: 0.0087046, 0.0192520, 0.0087729, 0.0194241, -0.0096450, 0.0094164
3: -0.0066545, -0.0019347, -0.0067892, -0.0019970, -0.0046575, 0.0048545
4: 0.0039107, 0.0054817, 0.0038680, 0.0054620, -0.0014575, 0.0015100
5: -0.0038354, -0.0003103, -0.0038607, -0.0003187, -0.0035167, 0.0035504
6: -0.0067229, -0.0050264, -0.0067614, -0.0050436, -0.0016794, 0.0017350
7: -0.0039236, -0.0005583, -0.0038466, -0.0004232, -0.0035004, 0.0032883
8: -0.0072211, -0.0002366, -0.0073273, -0.0002775, -0.0069436, 0.0070908
9: 1.0003281, 1.0020975, 1.0003380, 1.0020361, -0.0017080, 0.0017595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016114
time: 1.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016114
time: 1.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0048378, 0.0006956, -0.0055326, 0.0054858
1: 0.0033019, 0.0071539, 0.0032440, 0.0071585, -0.0038566, 0.0039099
2: 0.0090055, 0.0193522, 0.0090055, 0.0194335, -0.0093443, 0.0092552
3: -0.0067221, -0.0021260, -0.0067841, -0.0021219, -0.0046002, 0.0046581
4: 0.0038875, 0.0054204, 0.0038693, 0.0054216, -0.0014017, 0.0014380
5: -0.0038581, -0.0003803, -0.0038634, -0.0003846, -0.0034735, 0.0034831
6: -0.0067448, -0.0050854, -0.0067606, -0.0050845, -0.0016603, 0.0016752
7: -0.0037320, -0.0005139, -0.0037500, -0.0004389, -0.0032931, 0.0032361
8: -0.0072838, -0.0004284, -0.0073326, -0.0004285, -0.0068554, 0.0069042
9: 1.0003542, 1.0018717, 1.0003529, 1.0018735, -0.0015193, 0.0015188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016109, upper bound: 0.0016224
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016109, upper bound: 0.0016224
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0048353, 0.0006653, -0.0058395, 0.0054745
1: 0.0032942, 0.0074643, 0.0032701, 0.0071575, -0.0038633, 0.0041943
2: 0.0084066, 0.0193341, 0.0090103, 0.0193793, -0.0099240, 0.0092849
3: -0.0067131, -0.0017723, -0.0067545, -0.0021233, -0.0045898, 0.0049821
4: 0.0038907, 0.0055345, 0.0038790, 0.0054212, -0.0014341, 0.0015532
5: -0.0038575, -0.0002268, -0.0038486, -0.0003865, -0.0034710, 0.0036218
6: -0.0067417, -0.0049735, -0.0067507, -0.0050852, -0.0016565, 0.0017772
7: -0.0040409, -0.0005022, -0.0037494, -0.0004657, -0.0035752, 0.0032472
8: -0.0072705, -0.0000416, -0.0072976, -0.0004317, -0.0068387, 0.0072560
9: 1.0003084, 1.0023073, 1.0003531, 1.0018706, -0.0015622, 0.0019542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016108, upper bound: 0.0016224
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016108, upper bound: 0.0016224
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0049683, 0.0007223, -0.0055593, 0.0056164
1: 0.0033019, 0.0071539, 0.0032052, 0.0072674, -0.0039655, 0.0039487
2: 0.0090055, 0.0193522, 0.0087682, 0.0194774, -0.0093181, 0.0094221
3: -0.0067221, -0.0021260, -0.0068185, -0.0019956, -0.0047266, 0.0046925
4: 0.0038875, 0.0054204, 0.0038585, 0.0054625, -0.0014196, 0.0014251
5: -0.0038581, -0.0003803, -0.0038752, -0.0003169, -0.0035413, 0.0034949
6: -0.0067448, -0.0050854, -0.0067711, -0.0050429, -0.0017019, 0.0016857
7: -0.0037320, -0.0005139, -0.0038472, -0.0003972, -0.0033348, 0.0033333
8: -0.0072838, -0.0004284, -0.0073620, -0.0002743, -0.0070095, 0.0069336
9: 1.0003542, 1.0018717, 1.0003378, 1.0020391, -0.0016849, 0.0015339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016278
time: 1.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016278
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0049659, 0.0006930, -0.0058672, 0.0056050
1: 0.0032942, 0.0074643, 0.0032308, 0.0072664, -0.0039722, 0.0042335
2: 0.0084066, 0.0193341, 0.0087729, 0.0194241, -0.0098919, 0.0094493
3: -0.0067131, -0.0017723, -0.0067892, -0.0019970, -0.0047161, 0.0050169
4: 0.0038907, 0.0055345, 0.0038680, 0.0054620, -0.0014531, 0.0015345
5: -0.0038575, -0.0002268, -0.0038607, -0.0003187, -0.0035388, 0.0036339
6: -0.0067417, -0.0049735, -0.0067614, -0.0050436, -0.0016981, 0.0017879
7: -0.0040409, -0.0005022, -0.0038466, -0.0004232, -0.0036177, 0.0033444
8: -0.0072705, -0.0000416, -0.0073273, -0.0002775, -0.0069930, 0.0072858
9: 1.0003084, 1.0023073, 1.0003380, 1.0020361, -0.0017277, 0.0019693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016278
time: 1.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016278
time: 1.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0043260, 0.0006410, -0.0055985, 0.0048810
1: 0.0033580, 0.0072629, 0.0033055, 0.0067766, -0.0034186, 0.0039573
2: 0.0087894, 0.0191808, 0.0099497, 0.0193339, -0.0096267, 0.0083290
3: -0.0066432, -0.0020018, -0.0067276, -0.0025779, -0.0040652, 0.0047258
4: 0.0039155, 0.0054603, 0.0038874, 0.0052702, -0.0013028, 0.0014996
5: -0.0037978, -0.0003250, -0.0038326, -0.0006751, -0.0031227, 0.0035076
6: -0.0067153, -0.0050458, -0.0067435, -0.0052440, -0.0014712, 0.0016977
7: -0.0038447, -0.0005489, -0.0034426, -0.0004929, -0.0033518, 0.0028936
8: -0.0071707, -0.0002886, -0.0072679, -0.0010480, -0.0061227, 0.0069793
9: 1.0003382, 1.0020260, 1.0004048, 1.0012336, -0.0008954, 0.0016211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014298, upper bound: 0.0015692
time: 1.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014238, upper bound: 0.0015654
time: 1.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0044610, 0.0006786, -0.0056362, 0.0050160
1: 0.0033580, 0.0072629, 0.0032577, 0.0068947, -0.0035367, 0.0040052
2: 0.0087894, 0.0191808, 0.0097068, 0.0193948, -0.0096333, 0.0085334
3: -0.0066432, -0.0020018, -0.0067739, -0.0024416, -0.0042015, 0.0047721
4: 0.0039155, 0.0054603, 0.0038723, 0.0053146, -0.0013317, 0.0014973
5: -0.0037978, -0.0003250, -0.0038447, -0.0006047, -0.0031931, 0.0035197
6: -0.0067153, -0.0050458, -0.0067572, -0.0052002, -0.0015151, 0.0017114
7: -0.0038447, -0.0005489, -0.0035488, -0.0004399, -0.0034047, 0.0029999
8: -0.0071707, -0.0002886, -0.0073055, -0.0008898, -0.0062809, 0.0070169
9: 1.0003382, 1.0020260, 1.0003880, 1.0014049, -0.0010667, 0.0016379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014546, upper bound: 0.0015550
time: 1.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014546, upper bound: 0.0015550
time: 1.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0049575, 0.0005549, -0.0046036, 0.0005888, -0.0055463, 0.0051585
1: 0.0033580, 0.0072629, 0.0033049, 0.0070156, -0.0036576, 0.0039580
2: 0.0087894, 0.0191808, 0.0094494, 0.0192293, -0.0093997, 0.0087012
3: -0.0066432, -0.0020018, -0.0067024, -0.0023016, -0.0043416, 0.0047006
4: 0.0039155, 0.0054603, 0.0038970, 0.0053602, -0.0013692, 0.0014807
5: -0.0037978, -0.0003250, -0.0037904, -0.0005382, -0.0032596, 0.0034654
6: -0.0067153, -0.0050458, -0.0067300, -0.0051541, -0.0015612, 0.0016842
7: -0.0038447, -0.0005489, -0.0036721, -0.0004671, -0.0033776, 0.0031232
8: -0.0071707, -0.0002886, -0.0071967, -0.0007234, -0.0064474, 0.0069081
9: 1.0003382, 1.0020260, 1.0003704, 1.0015838, -0.0012456, 0.0016556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014357, upper bound: 0.0015543
time: 1.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014357, upper bound: 0.0015543
time: 1.48 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.47 seconds
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014303, upper bound: 0.0015692
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014243, upper bound: 0.0015654
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014551, upper bound: 0.0015550
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014551, upper bound: 0.0015550
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016093, upper bound: 0.0016098
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016122
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016228
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016228
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016228
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016113, upper bound: 0.0016228
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016284
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016284
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016284
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016284
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014545, upper bound: 0.0015543
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014545, upper bound: 0.0015543
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016089, upper bound: 0.0016092
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016089, upper bound: 0.0016092
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014087, upper bound: 0.0016092
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014087, upper bound: 0.0016092
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016114
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016114
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016114
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016222, upper bound: 0.0016114
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016109, upper bound: 0.0016224
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016109, upper bound: 0.0016224
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016108, upper bound: 0.0016224
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016108, upper bound: 0.0016224
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016278
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016278
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016278
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0016276, upper bound: 0.0016278
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014298, upper bound: 0.0015692
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014238, upper bound: 0.0015654
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014546, upper bound: 0.0015550
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014546, upper bound: 0.0015550
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014357, upper bound: 0.0015543
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.47
Output dim: 9, lower bound: -0.0014357, upper bound: 0.0015543

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1_B1

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0015113
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015986, upper bound: 0.0016414
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1_B2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0015113
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015986, upper bound: 0.0016414
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2_B1

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015187, upper bound: 0.0014621
time: 3.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015958, upper bound: 0.0015962
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2_B2

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015187, upper bound: 0.0014621
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015958, upper bound: 0.0015962
time: 1.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1_B1

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015401, upper bound: 0.0015133
time: 1.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016111, upper bound: 0.0016433
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1_B2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015401, upper bound: 0.0015133
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016111, upper bound: 0.0016433
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2_B1

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0014659
time: 2.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015987
time: 1.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2_B2

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0014659
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015987
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1_B1

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016013, upper bound: 0.0016726
time: 1.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1_B2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016013, upper bound: 0.0016726
time: 1.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2_B1

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015226, upper bound: 0.0014674
time: 2.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015980, upper bound: 0.0016091
time: 1.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2_B2

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015226, upper bound: 0.0014674
time: 1.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015980, upper bound: 0.0016091
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1_B1

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0015418
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016829
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0015418
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016829
time: 1.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2_B1

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013175, upper bound: 0.0014742
time: 2.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016152
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015375, upper bound: 0.0014742
time: 2.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016152
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0048278, 0.0005301, -0.0052221, 0.0054370
1: 0.0033514, 0.0070309, 0.0033941, 0.0071543, -0.0038029, 0.0036368
2: 0.0092692, 0.0192927, 0.0090253, 0.0191401, -0.0087650, 0.0091523
3: -0.0066708, -0.0022665, -0.0066109, -0.0021276, -0.0045432, 0.0043444
4: 0.0039042, 0.0053746, 0.0039253, 0.0054196, -0.0013867, 0.0013389
5: -0.0038431, -0.0004534, -0.0037876, -0.0003922, -0.0034509, 0.0033343
6: -0.0067294, -0.0051322, -0.0067061, -0.0050873, -0.0016421, 0.0015739
7: -0.0036275, -0.0005668, -0.0037476, -0.0005895, -0.0030380, 0.0031809
8: -0.0072464, -0.0005987, -0.0071428, -0.0004418, -0.0068047, 0.0065442
9: 1.0003716, 1.0016876, 1.0003535, 1.0018612, -0.0014896, 0.0013342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0015099
time: 1.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015983, upper bound: 0.0016405
time: 1.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0051470, 0.0005027, -0.0051947, 0.0057562
1: 0.0033514, 0.0070309, 0.0033933, 0.0074547, -0.0041033, 0.0036376
2: 0.0092692, 0.0192927, 0.0084569, 0.0190914, -0.0087399, 0.0097434
3: -0.0066708, -0.0022665, -0.0065925, -0.0017877, -0.0048831, 0.0043260
4: 0.0039042, 0.0053746, 0.0039324, 0.0055289, -0.0014976, 0.0013354
5: -0.0038431, -0.0004534, -0.0037792, -0.0002434, -0.0035997, 0.0033258
6: -0.0067294, -0.0051322, -0.0066975, -0.0049808, -0.0017486, 0.0015654
7: -0.0036275, -0.0005668, -0.0040536, -0.0005844, -0.0030431, 0.0034869
8: -0.0072464, -0.0005987, -0.0071131, -0.0000742, -0.0071722, 0.0065144
9: 1.0003716, 1.0016876, 1.0003088, 1.0022757, -0.0019041, 0.0013789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0015099
time: 2.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015983, upper bound: 0.0016405
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0048278, 0.0005301, -0.0055414, 0.0054153
1: 0.0033506, 0.0073251, 0.0033941, 0.0071543, -0.0038036, 0.0039310
2: 0.0087046, 0.0192520, 0.0090253, 0.0191401, -0.0093487, 0.0091284
3: -0.0066545, -0.0019347, -0.0066109, -0.0021276, -0.0045269, 0.0046762
4: 0.0039107, 0.0054817, 0.0039253, 0.0054196, -0.0013928, 0.0014557
5: -0.0038354, -0.0003103, -0.0037876, -0.0003922, -0.0034431, 0.0034774
6: -0.0067229, -0.0050264, -0.0067061, -0.0050873, -0.0016357, 0.0016797
7: -0.0039236, -0.0005583, -0.0037476, -0.0005895, -0.0033341, 0.0031893
8: -0.0072211, -0.0002366, -0.0071428, -0.0004418, -0.0067793, 0.0069063
9: 1.0003281, 1.0020975, 1.0003535, 1.0018612, -0.0015332, 0.0017440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015177, upper bound: 0.0014613
time: 1.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015955
time: 1.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0051470, 0.0005027, -0.0055140, 0.0057345
1: 0.0033506, 0.0073251, 0.0033933, 0.0074547, -0.0041040, 0.0039318
2: 0.0087046, 0.0192520, 0.0084569, 0.0190914, -0.0092197, 0.0096187
3: -0.0066545, -0.0019347, -0.0065925, -0.0017877, -0.0048668, 0.0046578
4: 0.0039107, 0.0054817, 0.0039324, 0.0055289, -0.0014333, 0.0013864
5: -0.0038354, -0.0003103, -0.0037792, -0.0002434, -0.0035920, 0.0034689
6: -0.0067229, -0.0050264, -0.0066975, -0.0049808, -0.0017421, 0.0016711
7: -0.0039236, -0.0005583, -0.0040536, -0.0005844, -0.0033392, 0.0034953
8: -0.0072211, -0.0002366, -0.0071131, -0.0000742, -0.0071468, 0.0068765
9: 1.0003281, 1.0020975, 1.0003088, 1.0022757, -0.0019476, 0.0017887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015177, upper bound: 0.0014613
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015955
time: 1.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0049575, 0.0005549, -0.0052470, 0.0055667
1: 0.0033514, 0.0070309, 0.0033580, 0.0072629, -0.0039115, 0.0036728
2: 0.0092692, 0.0192927, 0.0087894, 0.0191808, -0.0088093, 0.0093942
3: -0.0066708, -0.0022665, -0.0066432, -0.0020018, -0.0046690, 0.0043767
4: 0.0039042, 0.0053746, 0.0039155, 0.0054603, -0.0014232, 0.0013445
5: -0.0038431, -0.0004534, -0.0037978, -0.0003250, -0.0035181, 0.0033444
6: -0.0067294, -0.0051322, -0.0067153, -0.0050458, -0.0016835, 0.0015831
7: -0.0036275, -0.0005668, -0.0038447, -0.0005489, -0.0030786, 0.0032779
8: -0.0072464, -0.0005987, -0.0071707, -0.0002886, -0.0069578, 0.0065721
9: 1.0003716, 1.0016876, 1.0003382, 1.0020260, -0.0016544, 0.0013494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015397, upper bound: 0.0015119
time: 1.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016111, upper bound: 0.0016423
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0046920, 0.0006092, -0.0052958, 0.0005380, -0.0052301, 0.0059049
1: 0.0033514, 0.0070309, 0.0033464, 0.0075762, -0.0042248, 0.0036845
2: 0.0092692, 0.0192927, 0.0081890, 0.0191475, -0.0088133, 0.0100215
3: -0.0066708, -0.0022665, -0.0066350, -0.0016461, -0.0050248, 0.0043685
4: 0.0039042, 0.0053746, 0.0039192, 0.0055750, -0.0015418, 0.0013499
5: -0.0038431, -0.0004534, -0.0037958, -0.0001657, -0.0036775, 0.0033424
6: -0.0067294, -0.0051322, -0.0067104, -0.0049336, -0.0017957, 0.0015782
7: -0.0036275, -0.0005668, -0.0041612, -0.0005394, -0.0030881, 0.0035945
8: -0.0072464, -0.0005987, -0.0071495, 0.0000998, -0.0073462, 0.0065509
9: 1.0003716, 1.0016876, 1.0002913, 1.0024639, -0.0020924, 0.0013963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015397, upper bound: 0.0015119
time: 1.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016111, upper bound: 0.0016423
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0049575, 0.0005549, -0.0055662, 0.0055450
1: 0.0033506, 0.0073251, 0.0033580, 0.0072629, -0.0039122, 0.0039670
2: 0.0087046, 0.0192520, 0.0087894, 0.0191808, -0.0093930, 0.0093703
3: -0.0066545, -0.0019347, -0.0066432, -0.0020018, -0.0046526, 0.0047084
4: 0.0039107, 0.0054817, 0.0039155, 0.0054603, -0.0014293, 0.0014613
5: -0.0038354, -0.0003103, -0.0037978, -0.0003250, -0.0035103, 0.0034876
6: -0.0067229, -0.0050264, -0.0067153, -0.0050458, -0.0016771, 0.0016889
7: -0.0039236, -0.0005583, -0.0038447, -0.0005489, -0.0033747, 0.0032864
8: -0.0072211, -0.0002366, -0.0071707, -0.0002886, -0.0069325, 0.0069342
9: 1.0003281, 1.0020975, 1.0003382, 1.0020260, -0.0016979, 0.0017593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015251, upper bound: 0.0014649
time: 2.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015978
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0050113, 0.0005875, -0.0052958, 0.0005380, -0.0055493, 0.0058832
1: 0.0033506, 0.0073251, 0.0033464, 0.0075762, -0.0042255, 0.0039787
2: 0.0087046, 0.0192520, 0.0081890, 0.0191475, -0.0092844, 0.0098903
3: -0.0066545, -0.0019347, -0.0066350, -0.0016461, -0.0050084, 0.0047003
4: 0.0039107, 0.0054817, 0.0039192, 0.0055750, -0.0014723, 0.0013924
5: -0.0038354, -0.0003103, -0.0037958, -0.0001657, -0.0036697, 0.0034855
6: -0.0067229, -0.0050264, -0.0067104, -0.0049336, -0.0017893, 0.0016840
7: -0.0039236, -0.0005583, -0.0041612, -0.0005394, -0.0033842, 0.0036029
8: -0.0072211, -0.0002366, -0.0071495, 0.0000998, -0.0073209, 0.0069130
9: 1.0003281, 1.0020975, 1.0002913, 1.0024639, -0.0021359, 0.0018061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015250, upper bound: 0.0014649
time: 1.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013064, upper bound: 0.0015978
time: 1.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0048278, 0.0005301, -0.0053671, 0.0054759
1: 0.0033019, 0.0071539, 0.0033941, 0.0071543, -0.0038524, 0.0037598
2: 0.0090055, 0.0193522, 0.0090253, 0.0191401, -0.0090483, 0.0092348
3: -0.0067221, -0.0021260, -0.0066109, -0.0021276, -0.0045945, 0.0044849
4: 0.0038875, 0.0054204, 0.0039253, 0.0054196, -0.0013993, 0.0013833
5: -0.0038581, -0.0003803, -0.0037876, -0.0003922, -0.0034659, 0.0034073
6: -0.0067448, -0.0050854, -0.0067061, -0.0050873, -0.0016576, 0.0016207
7: -0.0037320, -0.0005139, -0.0037476, -0.0005895, -0.0031425, 0.0032338
8: -0.0072838, -0.0004284, -0.0071428, -0.0004418, -0.0068421, 0.0067145
9: 1.0003542, 1.0018717, 1.0003535, 1.0018612, -0.0015070, 0.0015182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015412, upper bound: 0.0015313
time: 1.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016008, upper bound: 0.0016710
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0051470, 0.0005027, -0.0053397, 0.0057951
1: 0.0033019, 0.0071539, 0.0033933, 0.0074547, -0.0041528, 0.0037606
2: 0.0090055, 0.0193522, 0.0084569, 0.0190914, -0.0090232, 0.0098259
3: -0.0067221, -0.0021260, -0.0065925, -0.0017877, -0.0049344, 0.0044665
4: 0.0038875, 0.0054204, 0.0039324, 0.0055289, -0.0015102, 0.0013798
5: -0.0038581, -0.0003803, -0.0037792, -0.0002434, -0.0036147, 0.0033989
6: -0.0067448, -0.0050854, -0.0066975, -0.0049808, -0.0017641, 0.0016121
7: -0.0037320, -0.0005139, -0.0040536, -0.0005844, -0.0031476, 0.0035397
8: -0.0072838, -0.0004284, -0.0071131, -0.0000742, -0.0072096, 0.0066847
9: 1.0003542, 1.0018717, 1.0003088, 1.0022757, -0.0019215, 0.0015630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015412, upper bound: 0.0015313
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016008, upper bound: 0.0016710
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0048278, 0.0005301, -0.0057042, 0.0054669
1: 0.0032942, 0.0074643, 0.0033941, 0.0071543, -0.0038601, 0.0040702
2: 0.0084066, 0.0193341, 0.0090253, 0.0191401, -0.0096801, 0.0092474
3: -0.0067131, -0.0017723, -0.0066109, -0.0021276, -0.0045855, 0.0048386
4: 0.0038907, 0.0055345, 0.0039253, 0.0054196, -0.0014147, 0.0015081
5: -0.0038575, -0.0002268, -0.0037876, -0.0003922, -0.0034653, 0.0035608
6: -0.0067417, -0.0049735, -0.0067061, -0.0050873, -0.0016544, 0.0017326
7: -0.0040409, -0.0005022, -0.0037476, -0.0005895, -0.0034514, 0.0032454
8: -0.0072705, -0.0000416, -0.0071428, -0.0004418, -0.0068287, 0.0071013
9: 1.0003084, 1.0023073, 1.0003535, 1.0018612, -0.0015528, 0.0019538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015216, upper bound: 0.0014663
time: 1.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015975, upper bound: 0.0016087
time: 1.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0051470, 0.0005027, -0.0056769, 0.0057862
1: 0.0032942, 0.0074643, 0.0033933, 0.0074547, -0.0041605, 0.0040710
2: 0.0084066, 0.0193341, 0.0084569, 0.0190914, -0.0095471, 0.0097302
3: -0.0067131, -0.0017723, -0.0065925, -0.0017877, -0.0049253, 0.0048202
4: 0.0038907, 0.0055345, 0.0039324, 0.0055289, -0.0014465, 0.0014339
5: -0.0038575, -0.0002268, -0.0037792, -0.0002434, -0.0036141, 0.0035524
6: -0.0067417, -0.0049735, -0.0066975, -0.0049808, -0.0017609, 0.0017240
7: -0.0040409, -0.0005022, -0.0040536, -0.0005844, -0.0034565, 0.0035514
8: -0.0072705, -0.0000416, -0.0071131, -0.0000742, -0.0071962, 0.0070715
9: 1.0003084, 1.0023073, 1.0003088, 1.0022757, -0.0019673, 0.0019985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015216, upper bound: 0.0014663
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015975, upper bound: 0.0016087
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0049575, 0.0005549, -0.0053919, 0.0056056
1: 0.0033019, 0.0071539, 0.0033580, 0.0072629, -0.0039610, 0.0037959
2: 0.0090055, 0.0193522, 0.0087894, 0.0191808, -0.0090173, 0.0094007
3: -0.0067221, -0.0021260, -0.0066432, -0.0020018, -0.0047203, 0.0045172
4: 0.0038875, 0.0054204, 0.0039155, 0.0054603, -0.0014170, 0.0013700
5: -0.0038581, -0.0003803, -0.0037978, -0.0003250, -0.0035331, 0.0034175
6: -0.0067448, -0.0050854, -0.0067153, -0.0050458, -0.0016990, 0.0016299
7: -0.0037320, -0.0005139, -0.0038447, -0.0005489, -0.0031830, 0.0033308
8: -0.0072838, -0.0004284, -0.0071707, -0.0002886, -0.0069952, 0.0067423
9: 1.0003542, 1.0018717, 1.0003382, 1.0020260, -0.0016718, 0.0015335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015573, upper bound: 0.0015404
time: 2.18 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016818
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0048370, 0.0006481, -0.0052958, 0.0005380, -0.0053750, 0.0059438
1: 0.0033019, 0.0071539, 0.0033464, 0.0075762, -0.0042743, 0.0038076
2: 0.0090055, 0.0193522, 0.0081890, 0.0191475, -0.0090178, 0.0100228
3: -0.0067221, -0.0021260, -0.0066350, -0.0016461, -0.0050761, 0.0045090
4: 0.0038875, 0.0054204, 0.0039192, 0.0055750, -0.0015302, 0.0013680
5: -0.0038581, -0.0003803, -0.0037958, -0.0001657, -0.0036925, 0.0034155
6: -0.0067448, -0.0050854, -0.0067104, -0.0049336, -0.0018112, 0.0016250
7: -0.0037320, -0.0005139, -0.0041612, -0.0005394, -0.0031926, 0.0036474
8: -0.0072838, -0.0004284, -0.0071495, 0.0000998, -0.0073837, 0.0067212
9: 1.0003542, 1.0018717, 1.0002913, 1.0024639, -0.0021098, 0.0015804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015573, upper bound: 0.0015404
time: 1.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016818
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0049575, 0.0005549, -0.0057291, 0.0055967
1: 0.0032942, 0.0074643, 0.0033580, 0.0072629, -0.0039687, 0.0041063
2: 0.0084066, 0.0193341, 0.0087894, 0.0191808, -0.0096410, 0.0094045
3: -0.0067131, -0.0017723, -0.0066432, -0.0020018, -0.0047112, 0.0048708
4: 0.0038907, 0.0055345, 0.0039155, 0.0054603, -0.0014246, 0.0014889
5: -0.0038575, -0.0002268, -0.0037978, -0.0003250, -0.0035325, 0.0035710
6: -0.0067417, -0.0049735, -0.0067153, -0.0050458, -0.0016958, 0.0017418
7: -0.0040409, -0.0005022, -0.0038447, -0.0005489, -0.0034920, 0.0033425
8: -0.0072705, -0.0000416, -0.0071707, -0.0002886, -0.0069819, 0.0071292
9: 1.0003084, 1.0023073, 1.0003382, 1.0020260, -0.0017176, 0.0019691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015374, upper bound: 0.0014737
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016146
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0051742, 0.0006392, -0.0052958, 0.0005380, -0.0057122, 0.0059349
1: 0.0032942, 0.0074643, 0.0033464, 0.0075762, -0.0042820, 0.0041179
2: 0.0084066, 0.0193341, 0.0081890, 0.0191475, -0.0095284, 0.0099179
3: -0.0067131, -0.0017723, -0.0066350, -0.0016461, -0.0050670, 0.0048627
4: 0.0038907, 0.0055345, 0.0039192, 0.0055750, -0.0014664, 0.0014204
5: -0.0038575, -0.0002268, -0.0037958, -0.0001657, -0.0036918, 0.0035690
6: -0.0067417, -0.0049735, -0.0067104, -0.0049336, -0.0018081, 0.0017369
7: -0.0040409, -0.0005022, -0.0041612, -0.0005394, -0.0035015, 0.0036591
8: -0.0072705, -0.0000416, -0.0071495, 0.0000998, -0.0073703, 0.0071080
9: 1.0003084, 1.0023073, 1.0002913, 1.0024639, -0.0021555, 0.0020159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012972, upper bound: 0.0014737
time: 2.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016146
time: 1.35 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.97 seconds
IS_A1_B1_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0015113
IS_A1_B1_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015986, upper bound: 0.0016414
IS_A1_B1_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0015113
IS_A1_B1_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015986, upper bound: 0.0016414
IS_A1_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015187, upper bound: 0.0014621
IS_A1_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015958, upper bound: 0.0015962
IS_A1_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015187, upper bound: 0.0014621
IS_A1_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015958, upper bound: 0.0015962
IS_A1_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015401, upper bound: 0.0015133
IS_A1_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016111, upper bound: 0.0016433
IS_A1_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015401, upper bound: 0.0015133
IS_A1_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016111, upper bound: 0.0016433
IS_A1_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0014659
IS_A1_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015987
IS_A1_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015252, upper bound: 0.0014659
IS_A1_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015987
IS_A1_B1_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
IS_A1_B1_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016013, upper bound: 0.0016726
IS_A1_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015422, upper bound: 0.0015334
IS_A1_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016013, upper bound: 0.0016726
IS_A1_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015226, upper bound: 0.0014674
IS_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015980, upper bound: 0.0016091
IS_A1_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015226, upper bound: 0.0014674
IS_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015980, upper bound: 0.0016091
IS_A1_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0015418
IS_A1_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016829
IS_A1_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015576, upper bound: 0.0015418
IS_A1_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016829
IS_A1_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0013175, upper bound: 0.0014742
IS_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016152
IS_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015375, upper bound: 0.0014742
IS_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016152
IS_A1_B2_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0015099
IS_A1_B2_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015983, upper bound: 0.0016405
IS_A1_B2_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015332, upper bound: 0.0015099
IS_A1_B2_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015983, upper bound: 0.0016405
IS_A1_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015177, upper bound: 0.0014613
IS_A1_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015955
IS_A1_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015177, upper bound: 0.0014613
IS_A1_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015953, upper bound: 0.0015955
IS_A1_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015397, upper bound: 0.0015119
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016111, upper bound: 0.0016423
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015397, upper bound: 0.0015119
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016111, upper bound: 0.0016423
IS_A1_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015251, upper bound: 0.0014649
IS_A1_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016087, upper bound: 0.0015978
IS_A1_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015250, upper bound: 0.0014649
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0013064, upper bound: 0.0015978
IS_A1_B2_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015412, upper bound: 0.0015313
IS_A1_B2_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016008, upper bound: 0.0016710
IS_A1_B2_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015412, upper bound: 0.0015313
IS_A1_B2_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016008, upper bound: 0.0016710
IS_A1_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015216, upper bound: 0.0014663
IS_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015975, upper bound: 0.0016087
IS_A1_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015216, upper bound: 0.0014663
IS_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015975, upper bound: 0.0016087
IS_A1_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015573, upper bound: 0.0015404
IS_A1_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016818
IS_A1_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015573, upper bound: 0.0015404
IS_A1_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016178, upper bound: 0.0016818
IS_A1_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0015374, upper bound: 0.0014737
IS_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016146
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0012972, upper bound: 0.0014737
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.97
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016146

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1_B1_A2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016040
time: 1.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016569
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0050113, 0.0005875, -0.0052180, 0.0056192
1: 0.0033547, 0.0069801, 0.0033506, 0.0073251, -0.0039703, 0.0036295
2: 0.0093798, 0.0192902, 0.0087046, 0.0192520, -0.0087111, 0.0094532
3: -0.0066698, -0.0023249, -0.0066545, -0.0019347, -0.0047351, 0.0043296
4: 0.0039045, 0.0053556, 0.0039107, 0.0054817, -0.0014388, 0.0012960
5: -0.0038415, -0.0004855, -0.0038354, -0.0003103, -0.0035312, 0.0033499
6: -0.0067290, -0.0051519, -0.0067229, -0.0050264, -0.0017026, 0.0015710
7: -0.0035825, -0.0005714, -0.0039236, -0.0005583, -0.0030242, 0.0033522
8: -0.0072445, -0.0006709, -0.0072211, -0.0002366, -0.0070080, 0.0065501
9: 1.0003786, 1.0016094, 1.0003281, 1.0020975, -0.0017189, 0.0012814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014682, upper bound: 0.0015568
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014682, upper bound: 0.0015568
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2_B1_A2

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015109, upper bound: 0.0015349
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015109, upper bound: 0.0015992
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2_B2_A2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014622, upper bound: 0.0015208
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014622, upper bound: 0.0015962
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0048370, 0.0006481, -0.0052786, 0.0054449
1: 0.0033547, 0.0069801, 0.0033019, 0.0071539, -0.0037992, 0.0036783
2: 0.0093798, 0.0192902, 0.0090055, 0.0193522, -0.0088096, 0.0091528
3: -0.0066698, -0.0023249, -0.0067221, -0.0021260, -0.0045438, 0.0043973
4: 0.0039045, 0.0053556, 0.0038875, 0.0054204, -0.0013664, 0.0012981
5: -0.0038415, -0.0004855, -0.0038581, -0.0003803, -0.0034612, 0.0033727
6: -0.0067290, -0.0051519, -0.0067448, -0.0050854, -0.0016436, 0.0015930
7: -0.0035825, -0.0005714, -0.0037320, -0.0005139, -0.0030687, 0.0031606
8: -0.0072445, -0.0006709, -0.0072838, -0.0004284, -0.0068161, 0.0066129
9: 1.0003786, 1.0016094, 1.0003542, 1.0018717, -0.0014931, 0.0012553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015623, upper bound: 0.0016119
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015623, upper bound: 0.0016602
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0051742, 0.0006392, -0.0052697, 0.0057821
1: 0.0033547, 0.0069801, 0.0032942, 0.0074643, -0.0041096, 0.0036860
2: 0.0093798, 0.0192902, 0.0084066, 0.0193341, -0.0088238, 0.0097846
3: -0.0066698, -0.0023249, -0.0067131, -0.0017723, -0.0048975, 0.0043882
4: 0.0039045, 0.0053556, 0.0038907, 0.0055345, -0.0014912, 0.0013151
5: -0.0038415, -0.0004855, -0.0038575, -0.0002268, -0.0036147, 0.0033720
6: -0.0067290, -0.0051519, -0.0067417, -0.0049735, -0.0017556, 0.0015898
7: -0.0035825, -0.0005714, -0.0040409, -0.0005022, -0.0030803, 0.0034695
8: -0.0072445, -0.0006709, -0.0072705, -0.0000416, -0.0072030, 0.0065995
9: 1.0003786, 1.0016094, 1.0003084, 1.0023073, -0.0019287, 0.0013011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0015596
time: 1.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0015596
time: 1.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2_B1_A2

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0015435
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0016027
time: 1.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0051742, 0.0006392, -0.0055873, 0.0057600
1: 0.0033546, 0.0072754, 0.0032942, 0.0074643, -0.0041097, 0.0039812
2: 0.0088181, 0.0192488, 0.0084066, 0.0193341, -0.0092919, 0.0096507
3: -0.0066533, -0.0019928, -0.0067131, -0.0017723, -0.0048809, 0.0047202
4: 0.0039111, 0.0054625, 0.0038907, 0.0055345, -0.0014182, 0.0013458
5: -0.0038333, -0.0003428, -0.0038575, -0.0002268, -0.0036065, 0.0035147
6: -0.0067225, -0.0050463, -0.0067417, -0.0049735, -0.0017490, 0.0016953
7: -0.0038787, -0.0005631, -0.0040409, -0.0005022, -0.0033765, 0.0034778
8: -0.0072188, -0.0003103, -0.0072705, -0.0000416, -0.0071772, 0.0069601
9: 1.0003350, 1.0020182, 1.0003084, 1.0023073, -0.0019723, 0.0017098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012607, upper bound: 0.0015253
time: 1.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014674, upper bound: 0.0015987
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1_B1_A2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016298
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016935
time: 1.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0050113, 0.0005875, -0.0053652, 0.0056581
1: 0.0033051, 0.0071064, 0.0033506, 0.0073251, -0.0040199, 0.0037558
2: 0.0091127, 0.0193497, 0.0087046, 0.0192520, -0.0089936, 0.0095360
3: -0.0067212, -0.0021812, -0.0066545, -0.0019347, -0.0047865, 0.0044732
4: 0.0038877, 0.0054023, 0.0039107, 0.0054817, -0.0014514, 0.0013402
5: -0.0038564, -0.0004109, -0.0038354, -0.0003103, -0.0035461, 0.0034244
6: -0.0067445, -0.0051042, -0.0067229, -0.0050264, -0.0017181, 0.0016187
7: -0.0036876, -0.0005181, -0.0039236, -0.0005583, -0.0031293, 0.0034055
8: -0.0072819, -0.0004982, -0.0072211, -0.0002366, -0.0070454, 0.0067229
9: 1.0003608, 1.0017968, 1.0003281, 1.0020975, -0.0017366, 0.0014688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0015775
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0016726
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2_B1_A2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0015413
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0016119
time: 2.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2_B2_A2

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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0015281
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0016091
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1_B1_A2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0016576
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0017067
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0051742, 0.0006392, -0.0054168, 0.0058210
1: 0.0033051, 0.0071064, 0.0032942, 0.0074643, -0.0041592, 0.0038123
2: 0.0091127, 0.0193497, 0.0084066, 0.0193341, -0.0090081, 0.0097713
3: -0.0067212, -0.0021812, -0.0067131, -0.0017723, -0.0049489, 0.0045318
4: 0.0038877, 0.0054023, 0.0038907, 0.0055345, -0.0014728, 0.0013256
5: -0.0038564, -0.0004109, -0.0038575, -0.0002268, -0.0036296, 0.0034466
6: -0.0067445, -0.0051042, -0.0067417, -0.0049735, -0.0017710, 0.0016375
7: -0.0036876, -0.0005181, -0.0040409, -0.0005022, -0.0031854, 0.0035228
8: -0.0072819, -0.0004982, -0.0072705, -0.0000416, -0.0072404, 0.0067723
9: 1.0003608, 1.0017968, 1.0003084, 1.0023073, -0.0019464, 0.0014884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014823, upper bound: 0.0015904
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014823, upper bound: 0.0016829
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2_B1_A2

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0015579
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0016184
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2_B2_A2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0015417
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0016152
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0048278, 0.0005301, -0.0051606, 0.0054358
1: 0.0033547, 0.0069801, 0.0033941, 0.0071543, -0.0037995, 0.0035860
2: 0.0093798, 0.0192902, 0.0090253, 0.0191401, -0.0086204, 0.0091487
3: -0.0066698, -0.0023249, -0.0066109, -0.0021276, -0.0045422, 0.0042860
4: 0.0039045, 0.0053556, 0.0039253, 0.0054196, -0.0013864, 0.0013070
5: -0.0038415, -0.0004855, -0.0037876, -0.0003922, -0.0034492, 0.0033022
6: -0.0067290, -0.0051519, -0.0067061, -0.0050873, -0.0016418, 0.0015542
7: -0.0035825, -0.0005714, -0.0037476, -0.0005895, -0.0029930, 0.0031762
8: -0.0072445, -0.0006709, -0.0071428, -0.0004418, -0.0068028, 0.0064719
9: 1.0003786, 1.0016094, 1.0003535, 1.0018612, -0.0014826, 0.0012560

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
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0016038
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0016563
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0051470, 0.0005027, -0.0051332, 0.0057550
1: 0.0033547, 0.0069801, 0.0033933, 0.0074547, -0.0040999, 0.0035868
2: 0.0093798, 0.0192902, 0.0084569, 0.0190914, -0.0085950, 0.0097398
3: -0.0066698, -0.0023249, -0.0065925, -0.0017877, -0.0048821, 0.0042676
4: 0.0039045, 0.0053556, 0.0039324, 0.0055289, -0.0014973, 0.0013016
5: -0.0038415, -0.0004855, -0.0037792, -0.0002434, -0.0035981, 0.0032937
6: -0.0067290, -0.0051519, -0.0066975, -0.0049808, -0.0017483, 0.0015456
7: -0.0035825, -0.0005714, -0.0040536, -0.0005844, -0.0029982, 0.0034822
8: -0.0072445, -0.0006709, -0.0071131, -0.0000742, -0.0071703, 0.0064422
9: 1.0003786, 1.0016094, 1.0003088, 1.0022757, -0.0018971, 0.0013007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014643, upper bound: 0.0015522
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014643, upper bound: 0.0016405
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0048278, 0.0005301, -0.0054782, 0.0054137
1: 0.0033546, 0.0072754, 0.0033941, 0.0071543, -0.0037997, 0.0038812
2: 0.0088181, 0.0192488, 0.0090253, 0.0191401, -0.0092026, 0.0091244
3: -0.0066533, -0.0019928, -0.0066109, -0.0021276, -0.0045257, 0.0046180
4: 0.0039111, 0.0054625, 0.0039253, 0.0054196, -0.0013924, 0.0014254
5: -0.0038333, -0.0003428, -0.0037876, -0.0003922, -0.0034411, 0.0034448
6: -0.0067225, -0.0050463, -0.0067061, -0.0050873, -0.0016353, 0.0016598
7: -0.0038787, -0.0005631, -0.0037476, -0.0005895, -0.0032892, 0.0031845
8: -0.0072188, -0.0003103, -0.0071428, -0.0004418, -0.0067770, 0.0068325
9: 1.0003350, 1.0020182, 1.0003535, 1.0018612, -0.0015262, 0.0016648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015099, upper bound: 0.0015344
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015099, upper bound: 0.0015987
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0051470, 0.0005027, -0.0054508, 0.0057329
1: 0.0033546, 0.0072754, 0.0033933, 0.0074547, -0.0041001, 0.0038821
2: 0.0088181, 0.0192488, 0.0084569, 0.0190914, -0.0090734, 0.0096146
3: -0.0066533, -0.0019928, -0.0065925, -0.0017877, -0.0048655, 0.0045996
4: 0.0039111, 0.0054625, 0.0039324, 0.0055289, -0.0014329, 0.0013521
5: -0.0038333, -0.0003428, -0.0037792, -0.0002434, -0.0035899, 0.0034363
6: -0.0067225, -0.0050463, -0.0066975, -0.0049808, -0.0017417, 0.0016512
7: -0.0038787, -0.0005631, -0.0040536, -0.0005844, -0.0032943, 0.0034905
8: -0.0072188, -0.0003103, -0.0071131, -0.0000742, -0.0071445, 0.0068028
9: 1.0003350, 1.0020182, 1.0003088, 1.0022757, -0.0019407, 0.0017095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012577, upper bound: 0.0015184
time: 1.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014587, upper bound: 0.0015184
time: 1.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0049575, 0.0005549, -0.0051855, 0.0055655
1: 0.0033547, 0.0069801, 0.0033580, 0.0072629, -0.0039081, 0.0036221
2: 0.0093798, 0.0192902, 0.0087894, 0.0191808, -0.0086580, 0.0093906
3: -0.0066698, -0.0023249, -0.0066432, -0.0020018, -0.0046680, 0.0043183
4: 0.0039045, 0.0053556, 0.0039155, 0.0054603, -0.0014229, 0.0013072
5: -0.0038415, -0.0004855, -0.0037978, -0.0003250, -0.0035164, 0.0033123
6: -0.0067290, -0.0051519, -0.0067153, -0.0050458, -0.0016832, 0.0015634
7: -0.0035825, -0.0005714, -0.0038447, -0.0005489, -0.0030336, 0.0032733
8: -0.0072445, -0.0006709, -0.0071707, -0.0002886, -0.0069559, 0.0064998
9: 1.0003786, 1.0016094, 1.0003382, 1.0020260, -0.0016474, 0.0012712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015623, upper bound: 0.0016116
time: 1.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015623, upper bound: 0.0016595
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046305, 0.0006080, -0.0052958, 0.0005380, -0.0051686, 0.0059037
1: 0.0033547, 0.0069801, 0.0033464, 0.0075762, -0.0042214, 0.0036338
2: 0.0093798, 0.0192902, 0.0081890, 0.0191475, -0.0086652, 0.0100180
3: -0.0066698, -0.0023249, -0.0066350, -0.0016461, -0.0050237, 0.0043101
4: 0.0039045, 0.0053556, 0.0039192, 0.0055750, -0.0015414, 0.0013130
5: -0.0038415, -0.0004855, -0.0037958, -0.0001657, -0.0036758, 0.0033103
6: -0.0067290, -0.0051519, -0.0067104, -0.0049336, -0.0017954, 0.0015585
7: -0.0035825, -0.0005714, -0.0041612, -0.0005394, -0.0030431, 0.0035899
8: -0.0072445, -0.0006709, -0.0071495, 0.0000998, -0.0073444, 0.0064786
9: 1.0003786, 1.0016094, 1.0002913, 1.0024639, -0.0020853, 0.0013181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014713, upper bound: 0.0015566
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014713, upper bound: 0.0016423
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0049575, 0.0005549, -0.0055031, 0.0055434
1: 0.0033546, 0.0072754, 0.0033580, 0.0072629, -0.0039083, 0.0039173
2: 0.0088181, 0.0192488, 0.0087894, 0.0191808, -0.0092402, 0.0093663
3: -0.0066533, -0.0019928, -0.0066432, -0.0020018, -0.0046514, 0.0046503
4: 0.0039111, 0.0054625, 0.0039155, 0.0054603, -0.0014289, 0.0014257
5: -0.0038333, -0.0003428, -0.0037978, -0.0003250, -0.0035083, 0.0034550
6: -0.0067225, -0.0050463, -0.0067153, -0.0050458, -0.0016767, 0.0016690
7: -0.0038787, -0.0005631, -0.0038447, -0.0005489, -0.0033297, 0.0032816
8: -0.0072188, -0.0003103, -0.0071707, -0.0002886, -0.0069302, 0.0068604
9: 1.0003350, 1.0020182, 1.0003382, 1.0020260, -0.0016910, 0.0016800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0015431
time: 1.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0016020
time: 1.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049481, 0.0005859, -0.0052958, 0.0005380, -0.0054862, 0.0058816
1: 0.0033546, 0.0072754, 0.0033464, 0.0075762, -0.0042216, 0.0039290
2: 0.0088181, 0.0192488, 0.0081890, 0.0191475, -0.0091332, 0.0098863
3: -0.0066533, -0.0019928, -0.0066350, -0.0016461, -0.0050072, 0.0046422
4: 0.0039111, 0.0054625, 0.0039192, 0.0055750, -0.0014719, 0.0013535
5: -0.0038333, -0.0003428, -0.0037958, -0.0001657, -0.0036676, 0.0034530
6: -0.0067225, -0.0050463, -0.0067104, -0.0049336, -0.0017889, 0.0016641
7: -0.0038787, -0.0005631, -0.0041612, -0.0005394, -0.0033392, 0.0035981
8: -0.0072188, -0.0003103, -0.0071495, 0.0000998, -0.0073186, 0.0068392
9: 1.0003350, 1.0020182, 1.0002913, 1.0024639, -0.0021290, 0.0017269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014657, upper bound: 0.0015242
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014657, upper bound: 0.0015978
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0048278, 0.0005301, -0.0053078, 0.0054746
1: 0.0033051, 0.0071064, 0.0033941, 0.0071543, -0.0038491, 0.0037123
2: 0.0091127, 0.0193497, 0.0090253, 0.0191401, -0.0089028, 0.0092315
3: -0.0067212, -0.0021812, -0.0066109, -0.0021276, -0.0045936, 0.0044297
4: 0.0038877, 0.0054023, 0.0039253, 0.0054196, -0.0013990, 0.0013512
5: -0.0038564, -0.0004109, -0.0037876, -0.0003922, -0.0034641, 0.0033767
6: -0.0067445, -0.0051042, -0.0067061, -0.0050873, -0.0016573, 0.0016019
7: -0.0036876, -0.0005181, -0.0037476, -0.0005895, -0.0030981, 0.0032295
8: -0.0072819, -0.0004982, -0.0071428, -0.0004418, -0.0068402, 0.0066447
9: 1.0003608, 1.0017968, 1.0003535, 1.0018612, -0.0015004, 0.0014434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015383, upper bound: 0.0016293
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015383, upper bound: 0.0016926
time: 1.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0051470, 0.0005027, -0.0052804, 0.0057939
1: 0.0033051, 0.0071064, 0.0033933, 0.0074547, -0.0041495, 0.0037131
2: 0.0091127, 0.0193497, 0.0084569, 0.0190914, -0.0088775, 0.0098226
3: -0.0067212, -0.0021812, -0.0065925, -0.0017877, -0.0049335, 0.0044113
4: 0.0038877, 0.0054023, 0.0039324, 0.0055289, -0.0015099, 0.0013458
5: -0.0038564, -0.0004109, -0.0037792, -0.0002434, -0.0036130, 0.0033682
6: -0.0067445, -0.0051042, -0.0066975, -0.0049808, -0.0017638, 0.0015933
7: -0.0036876, -0.0005181, -0.0040536, -0.0005844, -0.0031033, 0.0035355
8: -0.0072819, -0.0004982, -0.0071131, -0.0000742, -0.0072077, 0.0066149
9: 1.0003608, 1.0017968, 1.0003088, 1.0022757, -0.0019149, 0.0014881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014695, upper bound: 0.0015716
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014695, upper bound: 0.0016710
time: 1.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0051156, 0.0006376, -0.0048278, 0.0005301, -0.0056457, 0.0054654
1: 0.0032984, 0.0074150, 0.0033941, 0.0071543, -0.0038558, 0.0040209
2: 0.0085128, 0.0193310, 0.0090253, 0.0191401, -0.0095339, 0.0092434
3: -0.0067119, -0.0018286, -0.0066109, -0.0021276, -0.0045843, 0.0047822
4: 0.0038911, 0.0055163, 0.0039253, 0.0054196, -0.0014143, 0.0014748
5: -0.0038553, -0.0002577, -0.0037876, -0.0003922, -0.0034630, 0.0035299
6: -0.0067413, -0.0049922, -0.0067061, -0.0050873, -0.0016540, 0.0017139
7: -0.0039943, -0.0005063, -0.0037476, -0.0005895, -0.0034048, 0.0032413
8: -0.0072682, -0.0001106, -0.0071428, -0.0004418, -0.0068264, 0.0070322
9: 1.0003155, 1.0022328, 1.0003535, 1.0018612, -0.0015457, 0.0018793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015116, upper bound: 0.0015409
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0012965, upper bound: 0.0016115
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0051156, 0.0006376, -0.0051470, 0.0005027, -0.0056183, 0.0057846
1: 0.0032984, 0.0074150, 0.0033933, 0.0074547, -0.0041562, 0.0040217
2: 0.0085128, 0.0193310, 0.0084569, 0.0190914, -0.0094000, 0.0097263
3: -0.0067119, -0.0018286, -0.0065925, -0.0017877, -0.0049242, 0.0047638
4: 0.0038911, 0.0055163, 0.0039324, 0.0055289, -0.0014461, 0.0013982
5: -0.0038553, -0.0002577, -0.0037792, -0.0002434, -0.0036119, 0.0035215
6: -0.0067413, -0.0049922, -0.0066975, -0.0049808, -0.0017605, 0.0017053
7: -0.0039943, -0.0005063, -0.0040536, -0.0005844, -0.0034100, 0.0035473
8: -0.0072682, -0.0001106, -0.0071131, -0.0000742, -0.0071939, 0.0070025
9: 1.0003155, 1.0022328, 1.0003088, 1.0022757, -0.0019602, 0.0019240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0012633, upper bound: 0.0013045
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0012633, upper bound: 0.0016087
time: 1.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0049575, 0.0005549, -0.0053326, 0.0056044
1: 0.0033051, 0.0071064, 0.0033580, 0.0072629, -0.0039577, 0.0037484
2: 0.0091127, 0.0193497, 0.0087894, 0.0191808, -0.0088670, 0.0093972
3: -0.0067212, -0.0021812, -0.0066432, -0.0020018, -0.0047194, 0.0044619
4: 0.0038877, 0.0054023, 0.0039155, 0.0054603, -0.0014167, 0.0013337
5: -0.0038564, -0.0004109, -0.0037978, -0.0003250, -0.0035313, 0.0033869
6: -0.0067445, -0.0051042, -0.0067153, -0.0050458, -0.0016987, 0.0016111
7: -0.0036876, -0.0005181, -0.0038447, -0.0005489, -0.0031387, 0.0033265
8: -0.0072819, -0.0004982, -0.0071707, -0.0002886, -0.0069933, 0.0066726
9: 1.0003608, 1.0017968, 1.0003382, 1.0020260, -0.0016651, 0.0014586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013735, upper bound: 0.0016575
time: 1.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0017058
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0047777, 0.0006468, -0.0052958, 0.0005380, -0.0053157, 0.0059426
1: 0.0033051, 0.0071064, 0.0033464, 0.0075762, -0.0042710, 0.0037600
2: 0.0091127, 0.0193497, 0.0081890, 0.0191475, -0.0088697, 0.0100194
3: -0.0067212, -0.0021812, -0.0066350, -0.0016461, -0.0050752, 0.0044538
4: 0.0038877, 0.0054023, 0.0039192, 0.0055750, -0.0015298, 0.0013303
5: -0.0038564, -0.0004109, -0.0037958, -0.0001657, -0.0036907, 0.0033849
6: -0.0067445, -0.0051042, -0.0067104, -0.0049336, -0.0018109, 0.0016062
7: -0.0036876, -0.0005181, -0.0041612, -0.0005394, -0.0031482, 0.0036431
8: -0.0072819, -0.0004982, -0.0071495, 0.0000998, -0.0073818, 0.0066514
9: 1.0003608, 1.0017968, 1.0002913, 1.0024639, -0.0021031, 0.0015055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 216

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014806, upper bound: 0.0015869
time: 2.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0014806, upper bound: 0.0015869
time: 1.34 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.91 seconds
IS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016040
IS_A1_B1_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015347, upper bound: 0.0016569
IS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014682, upper bound: 0.0015568
IS_A1_B1_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014682, upper bound: 0.0015568
IS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015109, upper bound: 0.0015349
IS_A1_B1_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015109, upper bound: 0.0015992
IS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014622, upper bound: 0.0015208
IS_A1_B1_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014622, upper bound: 0.0015962
IS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015623, upper bound: 0.0016119
IS_A1_B1_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015623, upper bound: 0.0016602
IS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0015596
IS_A1_B1_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0015596
IS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0015435
IS_A1_B1_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0016027
IS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0012607, upper bound: 0.0015253
IS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014674, upper bound: 0.0015987
IS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016298
IS_A1_B1_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015390, upper bound: 0.0016935
IS_A1_B1_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0015775
IS_A1_B1_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014733, upper bound: 0.0016726
IS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0015413
IS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015125, upper bound: 0.0016119
IS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0015281
IS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014659, upper bound: 0.0016091
IS_A1_B1_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0016576
IS_A1_B1_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0017067
IS_A1_B1_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014823, upper bound: 0.0015904
IS_A1_B1_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014823, upper bound: 0.0016829
IS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0015579
IS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015392, upper bound: 0.0016184
IS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0015417
IS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014751, upper bound: 0.0016152
IS_A1_B2_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0016038
IS_A1_B2_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015341, upper bound: 0.0016563
IS_A1_B2_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014643, upper bound: 0.0015522
IS_A1_B2_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014643, upper bound: 0.0016405
IS_A1_B2_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015099, upper bound: 0.0015344
IS_A1_B2_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015099, upper bound: 0.0015987
IS_A1_B2_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0012577, upper bound: 0.0015184
IS_A1_B2_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014587, upper bound: 0.0015184
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015623, upper bound: 0.0016116
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015623, upper bound: 0.0016595
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014713, upper bound: 0.0015566
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014713, upper bound: 0.0016423
IS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0015431
IS_A1_B2_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015322, upper bound: 0.0016020
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014657, upper bound: 0.0015242
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014657, upper bound: 0.0015978
IS_A1_B2_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015383, upper bound: 0.0016293
IS_A1_B2_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015383, upper bound: 0.0016926
IS_A1_B2_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014695, upper bound: 0.0015716
IS_A1_B2_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014695, upper bound: 0.0016710
IS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015116, upper bound: 0.0015409
IS_A1_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0012965, upper bound: 0.0016115
IS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0012633, upper bound: 0.0013045
IS_A1_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0012633, upper bound: 0.0016087
IS_A1_B2_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0013735, upper bound: 0.0016575
IS_A1_B2_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0015741, upper bound: 0.0017058
IS_A1_B2_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014806, upper bound: 0.0015869
IS_A1_B2_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.91
Output dim: 9, lower bound: -0.0014806, upper bound: 0.0015869
IS_A1_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.91
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016146
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.91
Output dim: 9, lower bound: -0.0016147, upper bound: 0.0016146

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.85 + 597.48 = 601.33 seconds
