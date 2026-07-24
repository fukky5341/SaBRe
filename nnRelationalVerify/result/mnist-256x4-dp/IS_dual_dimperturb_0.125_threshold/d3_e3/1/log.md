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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0070121, 0.0080598, 0.0070121, 0.0080598, -0.0005255, 0.0005255)
1: (0.0023353, 0.0024867, 0.0023353, 0.0024867, -0.0000759, 0.0000759)
2: (0.0099038, 0.0104831, 0.0099038, 0.0104831, -0.0002905, 0.0002905)
3: (-0.0044375, -0.0038384, -0.0044375, -0.0038384, -0.0003005, 0.0003005)
4: (0.0001183, 0.0007669, 0.0001183, 0.0007669, -0.0003253, 0.0003253)
5: (0.0033876, 0.0040014, 0.0033876, 0.0040014, -0.0003079, 0.0003079)
6: (-0.0088592, -0.0064240, -0.0088592, -0.0064240, -0.0012215, 0.0012215)
7: (0.0061922, 0.0095087, 0.0061922, 0.0095087, -0.0016635, 0.0016635)
8: (0.9935758, 0.9959121, 0.9935758, 0.9959121, -0.0011718, 0.0011718)
9: (-0.0121765, -0.0100558, -0.0121765, -0.0100558, -0.0010637, 0.0010637)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.78 + 1.32 = 3.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0008845, upper bound: 0.0008845

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008542, upper bound: 0.0008177
time: 0.48 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008545, upper bound: 0.0008545
time: 0.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 8, lower bound: -0.0008542, upper bound: 0.0008177
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 8, lower bound: -0.0008545, upper bound: 0.0008545

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0070003, 0.0080161, 0.0070144, 0.0080475, -0.0005023, 0.0004697
1: 0.0023336, 0.0024804, 0.0023357, 0.0024849, -0.0000726, 0.0000679
2: 0.0099280, 0.0104896, 0.0099106, 0.0104818, -0.0002597, 0.0002777
3: -0.0044125, -0.0038316, -0.0044304, -0.0038397, -0.0002686, 0.0002872
4: 0.0001110, 0.0007398, 0.0001198, 0.0007593, -0.0003109, 0.0002908
5: 0.0034133, 0.0040083, 0.0033948, 0.0040000, -0.0002752, 0.0002943
6: -0.0087575, -0.0063966, -0.0088306, -0.0064294, -0.0010917, 0.0011675
7: 0.0061550, 0.0093703, 0.0061996, 0.0094698, -0.0015900, 0.0014868
8: 0.9935495, 0.9958145, 0.9935811, 0.9958847, -0.0011201, 0.0010474
9: -0.0120880, -0.0100320, -0.0121516, -0.0100606, -0.0009507, 0.0010167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008177, upper bound: 0.0008177
time: 0.53 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008177, upper bound: 0.0008177
time: 0.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0070131, 0.0080487, 0.0070121, 0.0080598, -0.0005251, 0.0004768
1: 0.0023355, 0.0024851, 0.0023353, 0.0024867, -0.0000759, 0.0000689
2: 0.0099100, 0.0104825, 0.0099038, 0.0104831, -0.0002636, 0.0002903
3: -0.0044311, -0.0038390, -0.0044375, -0.0038384, -0.0002726, 0.0003003
4: 0.0001190, 0.0007600, 0.0001183, 0.0007669, -0.0003250, 0.0002951
5: 0.0033942, 0.0040008, 0.0033876, 0.0040014, -0.0002793, 0.0003076
6: -0.0088333, -0.0064264, -0.0088592, -0.0064240, -0.0011082, 0.0012205
7: 0.0061955, 0.0094735, 0.0061922, 0.0095087, -0.0016622, 0.0015092
8: 0.9935782, 0.9958872, 0.9935758, 0.9959121, -0.0011709, 0.0010631
9: -0.0121539, -0.0100579, -0.0121765, -0.0100558, -0.0009650, 0.0010628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008177, upper bound: 0.0008542
time: 0.49 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0008177, upper bound: 0.0008545
time: 0.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.75 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 8, lower bound: -0.0008177, upper bound: 0.0008177
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 8, lower bound: -0.0008177, upper bound: 0.0008177
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 8, lower bound: -0.0008177, upper bound: 0.0008542
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 8, lower bound: -0.0008177, upper bound: 0.0008545

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0070003, 0.0080161, 0.0070003, 0.0080161, -0.0004656, 0.0004656
1: 0.0023336, 0.0024804, 0.0023336, 0.0024804, -0.0000673, 0.0000673
2: 0.0099280, 0.0104896, 0.0099280, 0.0104896, -0.0002574, 0.0002574
3: -0.0044125, -0.0038316, -0.0044125, -0.0038316, -0.0002662, 0.0002662
4: 0.0001110, 0.0007398, 0.0001110, 0.0007398, -0.0002882, 0.0002882
5: 0.0034133, 0.0040083, 0.0034133, 0.0040083, -0.0002727, 0.0002727
6: -0.0087575, -0.0063966, -0.0087575, -0.0063966, -0.0010821, 0.0010821
7: 0.0061550, 0.0093703, 0.0061550, 0.0093703, -0.0014738, 0.0014738
8: 0.9935495, 0.9958145, 0.9935495, 0.9958145, -0.0010381, 0.0010381
9: -0.0120880, -0.0100320, -0.0120880, -0.0100320, -0.0009424, 0.0009424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007071, upper bound: 0.0006053
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005592, upper bound: 0.0005592
time: 0.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0070003, 0.0080161, 0.0070131, 0.0080487, -0.0005166, 0.0004702
1: 0.0023336, 0.0024804, 0.0023355, 0.0024851, -0.0000746, 0.0000679
2: 0.0099280, 0.0104896, 0.0099100, 0.0104825, -0.0002600, 0.0002856
3: -0.0044125, -0.0038316, -0.0044311, -0.0038390, -0.0002689, 0.0002954
4: 0.0001110, 0.0007398, 0.0001190, 0.0007600, -0.0003198, 0.0002911
5: 0.0034133, 0.0040083, 0.0033942, 0.0040008, -0.0002754, 0.0003026
6: -0.0087575, -0.0063966, -0.0088333, -0.0064264, -0.0010928, 0.0012007
7: 0.0061550, 0.0093703, 0.0061955, 0.0094735, -0.0016352, 0.0014884
8: 0.9935495, 0.9958145, 0.9935782, 0.9958872, -0.0011519, 0.0010484
9: -0.0120880, -0.0100320, -0.0121539, -0.0100579, -0.0009517, 0.0010456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0006053, upper bound: 0.0007071
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005592, upper bound: 0.0005592
time: 0.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0070131, 0.0080487, 0.0070003, 0.0080161, -0.0004702, 0.0005166
1: 0.0023355, 0.0024851, 0.0023336, 0.0024804, -0.0000679, 0.0000746
2: 0.0099100, 0.0104825, 0.0099280, 0.0104896, -0.0002856, 0.0002600
3: -0.0044311, -0.0038390, -0.0044125, -0.0038316, -0.0002954, 0.0002689
4: 0.0001190, 0.0007600, 0.0001110, 0.0007398, -0.0002911, 0.0003198
5: 0.0033942, 0.0040008, 0.0034133, 0.0040083, -0.0003026, 0.0002754
6: -0.0088333, -0.0064264, -0.0087575, -0.0063966, -0.0012007, 0.0010928
7: 0.0061955, 0.0094735, 0.0061550, 0.0093703, -0.0014884, 0.0016352
8: 0.9935782, 0.9958872, 0.9935495, 0.9958145, -0.0010484, 0.0011519
9: -0.0121539, -0.0100579, -0.0120880, -0.0100320, -0.0010456, 0.0009517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007071, upper bound: 0.0006964
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005592, upper bound: 0.0006396
time: 0.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070131, 0.0080487, 0.0070131, 0.0080487, -0.0004763, 0.0004763
1: 0.0023355, 0.0024851, 0.0023355, 0.0024851, -0.0000688, 0.0000688
2: 0.0099100, 0.0104825, 0.0099100, 0.0104825, -0.0002633, 0.0002633
3: -0.0044311, -0.0038390, -0.0044311, -0.0038390, -0.0002723, 0.0002723
4: 0.0001190, 0.0007600, 0.0001190, 0.0007600, -0.0002948, 0.0002948
5: 0.0033942, 0.0040008, 0.0033942, 0.0040008, -0.0002790, 0.0002790
6: -0.0088333, -0.0064264, -0.0088333, -0.0064264, -0.0011070, 0.0011070
7: 0.0061955, 0.0094735, 0.0061955, 0.0094735, -0.0015077, 0.0015077
8: 0.9935782, 0.9958872, 0.9935782, 0.9958872, -0.0010620, 0.0010620
9: -0.0121539, -0.0100579, -0.0121539, -0.0100579, -0.0009641, 0.0009641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0007071, upper bound: 0.0007078
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005592, upper bound: 0.0006570
time: 0.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.77 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007071, upper bound: 0.0006053
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0005592, upper bound: 0.0005592
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0006053, upper bound: 0.0007071
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0005592, upper bound: 0.0005592
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007071, upper bound: 0.0006964
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0005592, upper bound: 0.0006396
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0007071, upper bound: 0.0007078
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.77
Output dim: 8, lower bound: -0.0005592, upper bound: 0.0006570

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.10 + 17.67 = 20.78 seconds
