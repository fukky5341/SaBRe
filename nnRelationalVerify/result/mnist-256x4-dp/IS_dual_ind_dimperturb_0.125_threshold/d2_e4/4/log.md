## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00058656


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0035823, 0.0047833, 0.0035823, 0.0047833, -0.0008772, 0.0008772)
1: (0.0018398, 0.0020134, 0.0018398, 0.0020134, -0.0001267, 0.0001267)
2: (0.0117153, 0.0123793, 0.0117153, 0.0123793, -0.0004850, 0.0004850)
3: (-0.0025640, -0.0018772, -0.0025640, -0.0018772, -0.0005016, 0.0005016)
4: (-0.0020048, -0.0012613, -0.0020048, -0.0012613, -0.0005430, 0.0005430)
5: (0.0053070, 0.0060106, 0.0053070, 0.0060106, -0.0005139, 0.0005139)
6: (-0.0012437, 0.0015478, -0.0012437, 0.0015478, -0.0020389, 0.0020389)
7: (-0.0046647, -0.0008628, -0.0046647, -0.0008628, -0.0027769, 0.0027769)
8: (0.9859279, 0.9886061, 0.9859279, 0.9886061, -0.0019561, 0.0019561)
9: (-0.0055446, -0.0031136, -0.0055446, -0.0031136, -0.0017756, 0.0017756)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 1.31 = 2.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0008000, upper bound: 0.0008001

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006579, upper bound: 0.0007652
time: 0.47 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007652, upper bound: 0.0007652
time: 0.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 8, lower bound: -0.0006579, upper bound: 0.0007652
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 8, lower bound: -0.0007652, upper bound: 0.0007652

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0037231, 0.0046933, 0.0036412, 0.0047822, -0.0007409, 0.0007185
1: 0.0018602, 0.0020004, 0.0018483, 0.0020132, -0.0001070, 0.0001038
2: 0.0117650, 0.0123015, 0.0117159, 0.0123467, -0.0003973, 0.0004096
3: -0.0025125, -0.0019577, -0.0025633, -0.0019109, -0.0004109, 0.0004237
4: -0.0019176, -0.0013170, -0.0019683, -0.0012620, -0.0004586, 0.0004448
5: 0.0053597, 0.0059281, 0.0053077, 0.0059760, -0.0004209, 0.0004340
6: -0.0010346, 0.0012206, -0.0012411, 0.0014109, -0.0016701, 0.0017221
7: -0.0042191, -0.0011476, -0.0044782, -0.0008665, -0.0023454, 0.0022745
8: 0.9862419, 0.9884055, 0.9860594, 0.9886035, -0.0016521, 0.0016022
9: -0.0053625, -0.0033986, -0.0055423, -0.0032329, -0.0014544, 0.0014997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006239, upper bound: 0.0006677
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006239, upper bound: 0.0007404
time: 0.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0036152, 0.0047818, 0.0035978, 0.0047827, -0.0006914, 0.0008630
1: 0.0018446, 0.0020131, 0.0018421, 0.0020133, -0.0000999, 0.0001247
2: 0.0117161, 0.0123611, 0.0117156, 0.0123707, -0.0004771, 0.0003823
3: -0.0025631, -0.0018960, -0.0025636, -0.0018861, -0.0004934, 0.0003953
4: -0.0019844, -0.0012622, -0.0019952, -0.0012617, -0.0004280, 0.0005342
5: 0.0053079, 0.0059913, 0.0053073, 0.0060015, -0.0005055, 0.0004050
6: -0.0012403, 0.0014713, -0.0012424, 0.0015117, -0.0020057, 0.0016070
7: -0.0045605, -0.0008675, -0.0046156, -0.0008647, -0.0021886, 0.0027317
8: 0.9860014, 0.9886028, 0.9859626, 0.9886048, -0.0015417, 0.0019242
9: -0.0055417, -0.0031802, -0.0055435, -0.0031450, -0.0017467, 0.0013994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007404, upper bound: 0.0006677
time: 0.48 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007404, upper bound: 0.0007404
time: 0.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 8, lower bound: -0.0006239, upper bound: 0.0006677
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 8, lower bound: -0.0006239, upper bound: 0.0007404
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 8, lower bound: -0.0007404, upper bound: 0.0006677
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 8, lower bound: -0.0007404, upper bound: 0.0007404

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037748, 0.0046930, 0.0037578, 0.0047355, -0.0006202, 0.0005923
1: 0.0018676, 0.0020003, 0.0018652, 0.0020064, -0.0000896, 0.0000856
2: 0.0117652, 0.0122729, 0.0117417, 0.0122823, -0.0003275, 0.0003429
3: -0.0025123, -0.0019873, -0.0025366, -0.0019776, -0.0003387, 0.0003546
4: -0.0018856, -0.0013172, -0.0018961, -0.0012909, -0.0003839, 0.0003666
5: 0.0053599, 0.0058978, 0.0053350, 0.0059077, -0.0003470, 0.0003633
6: -0.0010339, 0.0011004, -0.0011327, 0.0011398, -0.0013766, 0.0014415
7: -0.0040554, -0.0011486, -0.0041091, -0.0010141, -0.0019632, 0.0018748
8: 0.9863572, 0.9884048, 0.9863194, 0.9884995, -0.0013829, 0.0013207
9: -0.0053619, -0.0035032, -0.0054479, -0.0034689, -0.0011988, 0.0012553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0006269
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0006269
time: 0.47 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037335, 0.0046932, 0.0036668, 0.0047818, -0.0007341, 0.0005407
1: 0.0018617, 0.0020003, 0.0018520, 0.0020131, -0.0001061, 0.0000781
2: 0.0117651, 0.0122957, 0.0117161, 0.0123326, -0.0002990, 0.0004059
3: -0.0025124, -0.0019637, -0.0025631, -0.0019255, -0.0003092, 0.0004198
4: -0.0019112, -0.0013171, -0.0019525, -0.0012622, -0.0004544, 0.0003347
5: 0.0053598, 0.0059220, 0.0053079, 0.0059610, -0.0003168, 0.0004300
6: -0.0010343, 0.0011963, -0.0012403, 0.0013513, -0.0012568, 0.0017062
7: -0.0041859, -0.0011481, -0.0043971, -0.0008675, -0.0023238, 0.0017117
8: 0.9862651, 0.9884052, 0.9861165, 0.9886028, -0.0016369, 0.0012058
9: -0.0053622, -0.0034197, -0.0055416, -0.0032847, -0.0010945, 0.0014859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0007058
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0007058
time: 0.48 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036667, 0.0047816, 0.0037109, 0.0047363, -0.0005511, 0.0007420
1: 0.0018520, 0.0020131, 0.0018584, 0.0020066, -0.0000796, 0.0001072
2: 0.0117163, 0.0123326, 0.0117413, 0.0123082, -0.0004102, 0.0003047
3: -0.0025629, -0.0019255, -0.0025370, -0.0019508, -0.0004243, 0.0003151
4: -0.0019525, -0.0012624, -0.0019252, -0.0012905, -0.0003412, 0.0004593
5: 0.0053080, 0.0059611, 0.0053346, 0.0059352, -0.0004346, 0.0003228
6: -0.0012396, 0.0013516, -0.0011344, 0.0012488, -0.0017246, 0.0012810
7: -0.0043974, -0.0008684, -0.0042574, -0.0010118, -0.0017446, 0.0023487
8: 0.9861162, 0.9886021, 0.9862148, 0.9885011, -0.0012289, 0.0016545
9: -0.0055411, -0.0032845, -0.0054494, -0.0033740, -0.0015018, 0.0011155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0006269
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0006269
time: 0.49 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036245, 0.0047817, 0.0036188, 0.0047824, -0.0006856, 0.0007054
1: 0.0018459, 0.0020131, 0.0018451, 0.0020132, -0.0000991, 0.0001019
2: 0.0117162, 0.0123560, 0.0117158, 0.0123591, -0.0003900, 0.0003791
3: -0.0025630, -0.0019013, -0.0025634, -0.0018981, -0.0004033, 0.0003920
4: -0.0019787, -0.0012623, -0.0019822, -0.0012619, -0.0004244, 0.0004366
5: 0.0053079, 0.0059858, 0.0053075, 0.0059892, -0.0004132, 0.0004016
6: -0.0012400, 0.0014497, -0.0012416, 0.0014629, -0.0016395, 0.0015936
7: -0.0045310, -0.0008679, -0.0045490, -0.0008657, -0.0021703, 0.0022329
8: 0.9860221, 0.9886025, 0.9860094, 0.9886040, -0.0015288, 0.0015729
9: -0.0055414, -0.0031991, -0.0055428, -0.0031876, -0.0014278, 0.0013877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0007058
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0007058
time: 0.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0006269
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0006269
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0007058
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0007058
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0006269
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0006269
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0007058
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0007058

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038163, 0.0046919, 0.0037753, 0.0047351, -0.0005606, 0.0005678
1: 0.0018736, 0.0020001, 0.0018677, 0.0020064, -0.0000810, 0.0000820
2: 0.0117658, 0.0122500, 0.0117420, 0.0122726, -0.0003139, 0.0003099
3: -0.0025117, -0.0020110, -0.0025364, -0.0019876, -0.0003247, 0.0003206
4: -0.0018600, -0.0013179, -0.0018853, -0.0012912, -0.0003470, 0.0003515
5: 0.0053606, 0.0058735, 0.0053353, 0.0058975, -0.0003326, 0.0003284
6: -0.0010312, 0.0010040, -0.0011316, 0.0010992, -0.0013196, 0.0013030
7: -0.0039240, -0.0011523, -0.0040537, -0.0010156, -0.0017745, 0.0017972
8: 0.9864497, 0.9884022, 0.9863584, 0.9884984, -0.0012500, 0.0012660
9: -0.0053596, -0.0035872, -0.0054470, -0.0035043, -0.0011492, 0.0011347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0005437
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0006269
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038143, 0.0047060, 0.0037752, 0.0047351, -0.0005691, 0.0006146
1: 0.0018734, 0.0020022, 0.0018677, 0.0020064, -0.0000822, 0.0000888
2: 0.0117580, 0.0122510, 0.0117419, 0.0122726, -0.0003398, 0.0003146
3: -0.0025198, -0.0020099, -0.0025364, -0.0019875, -0.0003514, 0.0003254
4: -0.0018612, -0.0013092, -0.0018854, -0.0012912, -0.0003523, 0.0003805
5: 0.0053523, 0.0058746, 0.0053352, 0.0058975, -0.0003600, 0.0003334
6: -0.0010641, 0.0010085, -0.0011317, 0.0010994, -0.0014285, 0.0013228
7: -0.0039303, -0.0011075, -0.0040540, -0.0010154, -0.0018015, 0.0019455
8: 0.9864452, 0.9884337, 0.9863582, 0.9884986, -0.0012690, 0.0013705
9: -0.0053882, -0.0035832, -0.0054471, -0.0035041, -0.0012440, 0.0011519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0005437
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0006269
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037751, 0.0046920, 0.0036836, 0.0047813, -0.0006753, 0.0005161
1: 0.0018677, 0.0020002, 0.0018545, 0.0020131, -0.0000976, 0.0000746
2: 0.0117658, 0.0122727, 0.0117164, 0.0123233, -0.0002854, 0.0003734
3: -0.0025118, -0.0019874, -0.0025628, -0.0019351, -0.0002951, 0.0003862
4: -0.0018854, -0.0013178, -0.0019420, -0.0012626, -0.0004181, 0.0003195
5: 0.0053605, 0.0058976, 0.0053082, 0.0059512, -0.0003023, 0.0003956
6: -0.0010316, 0.0010997, -0.0012391, 0.0013122, -0.0011996, 0.0015697
7: -0.0040544, -0.0011518, -0.0043438, -0.0008692, -0.0021378, 0.0016338
8: 0.9863580, 0.9884026, 0.9861540, 0.9886016, -0.0015059, 0.0011509
9: -0.0053599, -0.0035039, -0.0055406, -0.0033188, -0.0010447, 0.0013670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0006163
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0007058
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037723, 0.0047062, 0.0036844, 0.0047814, -0.0006834, 0.0005727
1: 0.0018673, 0.0020022, 0.0018546, 0.0020131, -0.0000987, 0.0000827
2: 0.0117579, 0.0122743, 0.0117164, 0.0123228, -0.0003166, 0.0003779
3: -0.0025199, -0.0019858, -0.0025629, -0.0019356, -0.0003275, 0.0003908
4: -0.0018872, -0.0013090, -0.0019416, -0.0012625, -0.0004231, 0.0003545
5: 0.0053521, 0.0058993, 0.0053081, 0.0059507, -0.0003355, 0.0004004
6: -0.0010646, 0.0011062, -0.0012393, 0.0013104, -0.0013311, 0.0015885
7: -0.0040633, -0.0011069, -0.0043413, -0.0008689, -0.0021634, 0.0018129
8: 0.9863515, 0.9884341, 0.9861557, 0.9886017, -0.0015239, 0.0012770
9: -0.0053886, -0.0034982, -0.0055407, -0.0033204, -0.0011592, 0.0013833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0006163
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0007058
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037104, 0.0047803, 0.0037280, 0.0047358, -0.0004926, 0.0007191
1: 0.0018584, 0.0020129, 0.0018609, 0.0020065, -0.0000712, 0.0001039
2: 0.0117170, 0.0123085, 0.0117416, 0.0122987, -0.0003975, 0.0002724
3: -0.0025622, -0.0019505, -0.0025368, -0.0019605, -0.0004112, 0.0002817
4: -0.0019255, -0.0012632, -0.0019146, -0.0012908, -0.0003049, 0.0004451
5: 0.0053088, 0.0059355, 0.0053348, 0.0059252, -0.0004212, 0.0002886
6: -0.0012367, 0.0012499, -0.0011332, 0.0012090, -0.0016713, 0.0011450
7: -0.0042590, -0.0008724, -0.0042033, -0.0010133, -0.0015594, 0.0022761
8: 0.9862137, 0.9885994, 0.9862530, 0.9885000, -0.0010985, 0.0016034
9: -0.0055385, -0.0033730, -0.0054484, -0.0034087, -0.0014554, 0.0009971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0005253
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0005253
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037066, 0.0047987, 0.0037278, 0.0047358, -0.0004990, 0.0007360
1: 0.0018578, 0.0020156, 0.0018609, 0.0020065, -0.0000721, 0.0001063
2: 0.0117068, 0.0123106, 0.0117415, 0.0122989, -0.0004069, 0.0002759
3: -0.0025727, -0.0019483, -0.0025368, -0.0019604, -0.0004208, 0.0002853
4: -0.0019279, -0.0012518, -0.0019147, -0.0012907, -0.0003089, 0.0004556
5: 0.0052980, 0.0059377, 0.0053348, 0.0059253, -0.0004311, 0.0002923
6: -0.0012794, 0.0012589, -0.0011334, 0.0012095, -0.0017106, 0.0011599
7: -0.0042712, -0.0008143, -0.0042040, -0.0010132, -0.0015796, 0.0023296
8: 0.9862051, 0.9886403, 0.9862524, 0.9885002, -0.0011127, 0.0016410
9: -0.0055757, -0.0033652, -0.0054485, -0.0034082, -0.0014896, 0.0010101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0005253
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0005253
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036688, 0.0047804, 0.0036363, 0.0047819, -0.0006275, 0.0006826
1: 0.0018523, 0.0020129, 0.0018476, 0.0020131, -0.0000907, 0.0000986
2: 0.0117169, 0.0123315, 0.0117161, 0.0123495, -0.0003774, 0.0003469
3: -0.0025623, -0.0019267, -0.0025631, -0.0019081, -0.0003903, 0.0003588
4: -0.0019512, -0.0012631, -0.0019714, -0.0012622, -0.0003884, 0.0004226
5: 0.0053087, 0.0059599, 0.0053078, 0.0059789, -0.0003999, 0.0003676
6: -0.0012371, 0.0013466, -0.0012404, 0.0014222, -0.0015866, 0.0014585
7: -0.0043907, -0.0008720, -0.0044937, -0.0008674, -0.0019863, 0.0021609
8: 0.9861209, 0.9885997, 0.9860485, 0.9886028, -0.0013992, 0.0015222
9: -0.0055388, -0.0032888, -0.0055417, -0.0032230, -0.0013817, 0.0012701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0005948
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0005948
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036633, 0.0047988, 0.0036369, 0.0047819, -0.0006337, 0.0007035
1: 0.0018515, 0.0020156, 0.0018477, 0.0020132, -0.0000916, 0.0001016
2: 0.0117067, 0.0123345, 0.0117161, 0.0123491, -0.0003889, 0.0003504
3: -0.0025728, -0.0019235, -0.0025632, -0.0019084, -0.0004022, 0.0003624
4: -0.0019546, -0.0012517, -0.0019710, -0.0012622, -0.0003923, 0.0004355
5: 0.0052979, 0.0059631, 0.0053078, 0.0059786, -0.0004121, 0.0003712
6: -0.0012798, 0.0013595, -0.0012406, 0.0014208, -0.0016350, 0.0014729
7: -0.0044083, -0.0008137, -0.0044918, -0.0008672, -0.0020060, 0.0022268
8: 0.9861085, 0.9886407, 0.9860497, 0.9886031, -0.0014131, 0.0015686
9: -0.0055761, -0.0032776, -0.0055419, -0.0032242, -0.0014238, 0.0012827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0005948
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0005948
time: 0.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.57 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0005437
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0006269
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0005437
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0006269
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0006163
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0007058
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0006163
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0005948, upper bound: 0.0007058
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0005253
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0005253
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0005253
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0005253
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0005948
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0006955, upper bound: 0.0005948
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0005948
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.57
Output dim: 8, lower bound: -0.0007058, upper bound: 0.0005948

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0038163, 0.0046919, 0.0037428, 0.0047347, -0.0005603, 0.0006179
1: 0.0018736, 0.0020001, 0.0018630, 0.0020063, -0.0000809, 0.0000893
2: 0.0117658, 0.0122500, 0.0117422, 0.0122906, -0.0003416, 0.0003098
3: -0.0025117, -0.0020110, -0.0025361, -0.0019690, -0.0003533, 0.0003204
4: -0.0018600, -0.0013179, -0.0019054, -0.0012914, -0.0003468, 0.0003825
5: 0.0053606, 0.0058735, 0.0053355, 0.0059165, -0.0003620, 0.0003282
6: -0.0010312, 0.0010040, -0.0011307, 0.0011748, -0.0014362, 0.0013023
7: -0.0039240, -0.0011523, -0.0041566, -0.0010168, -0.0017736, 0.0019559
8: 0.9864497, 0.9884022, 0.9862859, 0.9884976, -0.0012494, 0.0013778
9: -0.0053596, -0.0035872, -0.0054462, -0.0034385, -0.0012507, 0.0011341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006269
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006269
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0038143, 0.0047060, 0.0038532, 0.0046422, -0.0004565, 0.0005455
1: 0.0018734, 0.0020022, 0.0018790, 0.0019930, -0.0000660, 0.0000788
2: 0.0117580, 0.0122510, 0.0117933, 0.0122295, -0.0003016, 0.0002524
3: -0.0025198, -0.0020099, -0.0024833, -0.0020321, -0.0003119, 0.0002611
4: -0.0018612, -0.0013092, -0.0018371, -0.0013487, -0.0002826, 0.0003377
5: 0.0053523, 0.0058746, 0.0053896, 0.0058518, -0.0003196, 0.0002674
6: -0.0010641, 0.0010085, -0.0009158, 0.0009181, -0.0012679, 0.0010611
7: -0.0039303, -0.0011075, -0.0038071, -0.0013094, -0.0014452, 0.0017268
8: 0.9864452, 0.9884337, 0.9865320, 0.9882915, -0.0010180, 0.0012164
9: -0.0053882, -0.0035832, -0.0052591, -0.0036620, -0.0011042, 0.0009241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0005437
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0005437
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0038143, 0.0047060, 0.0037420, 0.0047347, -0.0005688, 0.0006556
1: 0.0018734, 0.0020022, 0.0018629, 0.0020063, -0.0000822, 0.0000947
2: 0.0117580, 0.0122510, 0.0117422, 0.0122910, -0.0003624, 0.0003145
3: -0.0025198, -0.0020099, -0.0025362, -0.0019685, -0.0003749, 0.0003253
4: -0.0018612, -0.0013092, -0.0019059, -0.0012914, -0.0003521, 0.0004058
5: 0.0053523, 0.0058746, 0.0053355, 0.0059170, -0.0003840, 0.0003332
6: -0.0010641, 0.0010085, -0.0011308, 0.0011766, -0.0015237, 0.0013221
7: -0.0039303, -0.0011075, -0.0041591, -0.0010166, -0.0018006, 0.0020752
8: 0.9864452, 0.9884337, 0.9862842, 0.9884977, -0.0012684, 0.0014618
9: -0.0053882, -0.0035832, -0.0054463, -0.0034369, -0.0013269, 0.0011513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006269
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006269
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037751, 0.0046920, 0.0037664, 0.0046925, -0.0005791, 0.0004405
1: 0.0018677, 0.0020002, 0.0018664, 0.0020002, -0.0000837, 0.0000636
2: 0.0117658, 0.0122727, 0.0117655, 0.0122775, -0.0002435, 0.0003202
3: -0.0025118, -0.0019874, -0.0025120, -0.0019825, -0.0002519, 0.0003312
4: -0.0018854, -0.0013178, -0.0018908, -0.0013175, -0.0003585, 0.0002727
5: 0.0053605, 0.0058976, 0.0053602, 0.0059027, -0.0002580, 0.0003393
6: -0.0010316, 0.0010997, -0.0010327, 0.0011198, -0.0010238, 0.0013461
7: -0.0040544, -0.0011518, -0.0040817, -0.0011503, -0.0018332, 0.0013943
8: 0.9863580, 0.9884026, 0.9863386, 0.9884036, -0.0012914, 0.0009822
9: -0.0053599, -0.0035039, -0.0053608, -0.0034864, -0.0008915, 0.0011722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006163
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0005437
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037751, 0.0046920, 0.0036562, 0.0047810, -0.0006751, 0.0005659
1: 0.0018677, 0.0020002, 0.0018505, 0.0020130, -0.0000975, 0.0000817
2: 0.0117658, 0.0122727, 0.0117166, 0.0123384, -0.0003128, 0.0003733
3: -0.0025118, -0.0019874, -0.0025626, -0.0019195, -0.0003236, 0.0003861
4: -0.0018854, -0.0013178, -0.0019590, -0.0012628, -0.0004179, 0.0003503
5: 0.0053605, 0.0058976, 0.0053083, 0.0059672, -0.0003315, 0.0003955
6: -0.0010316, 0.0010997, -0.0012384, 0.0013759, -0.0013152, 0.0015692
7: -0.0040544, -0.0011518, -0.0044305, -0.0008701, -0.0021371, 0.0017912
8: 0.9863580, 0.9884026, 0.9860929, 0.9886009, -0.0015054, 0.0012618
9: -0.0053599, -0.0035039, -0.0055400, -0.0032633, -0.0011453, 0.0013665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0007058
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006269
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037723, 0.0047062, 0.0037664, 0.0046926, -0.0005870, 0.0005033
1: 0.0018673, 0.0020022, 0.0018664, 0.0020002, -0.0000848, 0.0000727
2: 0.0117579, 0.0122743, 0.0117655, 0.0122775, -0.0002783, 0.0003246
3: -0.0025199, -0.0019858, -0.0025121, -0.0019825, -0.0002878, 0.0003357
4: -0.0018872, -0.0013090, -0.0018908, -0.0013175, -0.0003634, 0.0003115
5: 0.0053521, 0.0058993, 0.0053601, 0.0059027, -0.0002948, 0.0003439
6: -0.0010646, 0.0011062, -0.0010328, 0.0011198, -0.0011698, 0.0013644
7: -0.0040633, -0.0011069, -0.0040818, -0.0011501, -0.0018582, 0.0015931
8: 0.9863515, 0.9884341, 0.9863386, 0.9884037, -0.0013090, 0.0011222
9: -0.0053886, -0.0034982, -0.0053610, -0.0034863, -0.0010187, 0.0011882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006163
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0005437
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037723, 0.0047062, 0.0036550, 0.0047811, -0.0006832, 0.0006134
1: 0.0018673, 0.0020022, 0.0018503, 0.0020130, -0.0000987, 0.0000886
2: 0.0117579, 0.0122743, 0.0117165, 0.0123391, -0.0003391, 0.0003777
3: -0.0025199, -0.0019858, -0.0025627, -0.0019188, -0.0003507, 0.0003907
4: -0.0018872, -0.0013090, -0.0019598, -0.0012627, -0.0004229, 0.0003797
5: 0.0053521, 0.0058993, 0.0053083, 0.0059680, -0.0003593, 0.0004002
6: -0.0010646, 0.0011062, -0.0012385, 0.0013788, -0.0014257, 0.0015880
7: -0.0040633, -0.0011069, -0.0044345, -0.0008699, -0.0021627, 0.0019417
8: 0.9863515, 0.9884341, 0.9860901, 0.9886011, -0.0015234, 0.0013677
9: -0.0053886, -0.0034982, -0.0055401, -0.0032608, -0.0012415, 0.0013829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0007058
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006269
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037104, 0.0047803, 0.0038553, 0.0046422, -0.0005783, 0.0005879
1: 0.0018584, 0.0020129, 0.0018793, 0.0019930, -0.0000835, 0.0000849
2: 0.0117170, 0.0123085, 0.0117933, 0.0122283, -0.0003250, 0.0003197
3: -0.0025622, -0.0019505, -0.0024832, -0.0020333, -0.0003361, 0.0003307
4: -0.0019255, -0.0012632, -0.0018358, -0.0013487, -0.0003580, 0.0003639
5: 0.0053088, 0.0059355, 0.0053897, 0.0058506, -0.0003444, 0.0003388
6: -0.0012367, 0.0012499, -0.0009157, 0.0009131, -0.0013663, 0.0013441
7: -0.0042590, -0.0008724, -0.0038003, -0.0013096, -0.0018306, 0.0018608
8: 0.9862137, 0.9885994, 0.9865368, 0.9882914, -0.0012895, 0.0013108
9: -0.0055385, -0.0033730, -0.0052589, -0.0036663, -0.0011899, 0.0011705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037104, 0.0047803, 0.0037428, 0.0047347, -0.0004923, 0.0005380
1: 0.0018584, 0.0020129, 0.0018630, 0.0020063, -0.0000711, 0.0000777
2: 0.0117170, 0.0123085, 0.0117422, 0.0122906, -0.0002974, 0.0002722
3: -0.0025622, -0.0019505, -0.0025361, -0.0019690, -0.0003076, 0.0002815
4: -0.0019255, -0.0012632, -0.0019054, -0.0012914, -0.0003047, 0.0003330
5: 0.0053088, 0.0059355, 0.0053355, 0.0059165, -0.0003151, 0.0002884
6: -0.0012367, 0.0012499, -0.0011307, 0.0011748, -0.0012504, 0.0011443
7: -0.0042590, -0.0008724, -0.0041566, -0.0010168, -0.0015584, 0.0017029
8: 0.9862137, 0.9885994, 0.9862859, 0.9884976, -0.0010978, 0.0011996
9: -0.0055385, -0.0033730, -0.0054462, -0.0034385, -0.0010889, 0.0009965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037066, 0.0047987, 0.0038532, 0.0046422, -0.0005875, 0.0006149
1: 0.0018578, 0.0020156, 0.0018790, 0.0019930, -0.0000849, 0.0000888
2: 0.0117068, 0.0123106, 0.0117933, 0.0122295, -0.0003400, 0.0003248
3: -0.0025727, -0.0019483, -0.0024833, -0.0020321, -0.0003516, 0.0003359
4: -0.0019279, -0.0012518, -0.0018371, -0.0013487, -0.0003637, 0.0003806
5: 0.0052980, 0.0059377, 0.0053896, 0.0058518, -0.0003602, 0.0003442
6: -0.0012794, 0.0012589, -0.0009158, 0.0009181, -0.0014292, 0.0013656
7: -0.0042712, -0.0008143, -0.0038071, -0.0013094, -0.0018598, 0.0019465
8: 0.9862051, 0.9886403, 0.9865320, 0.9882915, -0.0013101, 0.0013711
9: -0.0055757, -0.0033652, -0.0052591, -0.0036620, -0.0012446, 0.0011892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005253
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005253
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037066, 0.0047987, 0.0037420, 0.0047347, -0.0004987, 0.0005890
1: 0.0018578, 0.0020156, 0.0018629, 0.0020063, -0.0000720, 0.0000851
2: 0.0117068, 0.0123106, 0.0117422, 0.0122910, -0.0003256, 0.0002757
3: -0.0025727, -0.0019483, -0.0025362, -0.0019685, -0.0003368, 0.0002852
4: -0.0019279, -0.0012518, -0.0019059, -0.0012914, -0.0003087, 0.0003646
5: 0.0052980, 0.0059377, 0.0053355, 0.0059170, -0.0003450, 0.0002921
6: -0.0012794, 0.0012589, -0.0011308, 0.0011766, -0.0013689, 0.0011591
7: -0.0042712, -0.0008143, -0.0041591, -0.0010166, -0.0015786, 0.0018643
8: 0.9862051, 0.9886403, 0.9862842, 0.9884977, -0.0011120, 0.0013133
9: -0.0055757, -0.0033652, -0.0054463, -0.0034369, -0.0011921, 0.0010094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005253
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005253
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0036688, 0.0047804, 0.0037664, 0.0046925, -0.0007026, 0.0005518
1: 0.0018523, 0.0020129, 0.0018664, 0.0020002, -0.0001015, 0.0000797
2: 0.0117169, 0.0123315, 0.0117655, 0.0122775, -0.0003051, 0.0003884
3: -0.0025623, -0.0019267, -0.0025120, -0.0019825, -0.0003155, 0.0004017
4: -0.0019512, -0.0012631, -0.0018908, -0.0013175, -0.0004349, 0.0003416
5: 0.0053087, 0.0059599, 0.0053602, 0.0059027, -0.0003232, 0.0004116
6: -0.0012371, 0.0013466, -0.0010327, 0.0011198, -0.0012826, 0.0016329
7: -0.0043907, -0.0008720, -0.0040817, -0.0011503, -0.0022239, 0.0017467
8: 0.9861209, 0.9885997, 0.9863386, 0.9884036, -0.0015666, 0.0012304
9: -0.0055388, -0.0032888, -0.0053608, -0.0034864, -0.0011169, 0.0014220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005948
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005253
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0036688, 0.0047804, 0.0036562, 0.0047810, -0.0006271, 0.0004837
1: 0.0018523, 0.0020129, 0.0018505, 0.0020130, -0.0000906, 0.0000699
2: 0.0117169, 0.0123315, 0.0117166, 0.0123384, -0.0002674, 0.0003467
3: -0.0025623, -0.0019267, -0.0025626, -0.0019195, -0.0002766, 0.0003586
4: -0.0019512, -0.0012631, -0.0019590, -0.0012628, -0.0003882, 0.0002994
5: 0.0053087, 0.0059599, 0.0053083, 0.0059672, -0.0002833, 0.0003673
6: -0.0012371, 0.0013466, -0.0012384, 0.0013759, -0.0011242, 0.0014575
7: -0.0043907, -0.0008720, -0.0044305, -0.0008701, -0.0019849, 0.0015311
8: 0.9861209, 0.9885997, 0.9860929, 0.9886009, -0.0013982, 0.0010785
9: -0.0055388, -0.0032888, -0.0055400, -0.0032633, -0.0009790, 0.0012692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005948
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005253
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036633, 0.0047988, 0.0037664, 0.0046926, -0.0007116, 0.0005823
1: 0.0018515, 0.0020156, 0.0018664, 0.0020002, -0.0001028, 0.0000841
2: 0.0117067, 0.0123345, 0.0117655, 0.0122775, -0.0003219, 0.0003934
3: -0.0025728, -0.0019235, -0.0025121, -0.0019825, -0.0003330, 0.0004069
4: -0.0019546, -0.0012517, -0.0018908, -0.0013175, -0.0004405, 0.0003605
5: 0.0052979, 0.0059631, 0.0053601, 0.0059027, -0.0003411, 0.0004168
6: -0.0012798, 0.0013595, -0.0010328, 0.0011198, -0.0013535, 0.0016539
7: -0.0044083, -0.0008137, -0.0040818, -0.0011501, -0.0022525, 0.0018433
8: 0.9861085, 0.9886407, 0.9863386, 0.9884037, -0.0015867, 0.0012984
9: -0.0055761, -0.0032776, -0.0053610, -0.0034863, -0.0011786, 0.0014403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005948
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005253
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036633, 0.0047988, 0.0036550, 0.0047811, -0.0006333, 0.0005437
1: 0.0018515, 0.0020156, 0.0018503, 0.0020130, -0.0000915, 0.0000786
2: 0.0117067, 0.0123345, 0.0117165, 0.0123391, -0.0003006, 0.0003501
3: -0.0025728, -0.0019235, -0.0025627, -0.0019188, -0.0003109, 0.0003621
4: -0.0019546, -0.0012517, -0.0019598, -0.0012627, -0.0003920, 0.0003366
5: 0.0052979, 0.0059631, 0.0053083, 0.0059680, -0.0003185, 0.0003710
6: -0.0012798, 0.0013595, -0.0012385, 0.0013788, -0.0012637, 0.0014719
7: -0.0044083, -0.0008137, -0.0044345, -0.0008699, -0.0020046, 0.0017211
8: 0.9861085, 0.9886407, 0.9860901, 0.9886011, -0.0014121, 0.0012124
9: -0.0055761, -0.0032776, -0.0055401, -0.0032608, -0.0011005, 0.0012818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005948
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005253
time: 0.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.67 seconds
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006269
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006269
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0005437
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0005437
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006269
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006269
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006163
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0005437
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0007058
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006269
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006163
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0005437
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0007058
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006269
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005253
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005253
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005253
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005253
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005948
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005253
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005948
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005253
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005948
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005253
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005948
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005253

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0037428, 0.0047347, -0.0004918, 0.0005394
1: 0.0018830, 0.0019929, 0.0018630, 0.0020063, -0.0000711, 0.0000779
2: 0.0117937, 0.0122142, 0.0117422, 0.0122906, -0.0002982, 0.0002719
3: -0.0024829, -0.0020480, -0.0025361, -0.0019690, -0.0003084, 0.0002812
4: -0.0018199, -0.0013491, -0.0019054, -0.0012914, -0.0003045, 0.0003339
5: 0.0053901, 0.0058356, 0.0053355, 0.0059165, -0.0003160, 0.0002881
6: -0.0009141, 0.0008537, -0.0011307, 0.0011748, -0.0012538, 0.0011432
7: -0.0037193, -0.0013118, -0.0041566, -0.0010168, -0.0015569, 0.0017075
8: 0.9865939, 0.9882898, 0.9862859, 0.9884976, -0.0010967, 0.0012028
9: -0.0052575, -0.0037181, -0.0054462, -0.0034385, -0.0010918, 0.0009955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006253
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006269
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037917, 0.0046918, 0.0037428, 0.0047347, -0.0006051, 0.0006179
1: 0.0018701, 0.0020001, 0.0018630, 0.0020063, -0.0000874, 0.0000893
2: 0.0117659, 0.0122635, 0.0117422, 0.0122906, -0.0003416, 0.0003345
3: -0.0025116, -0.0019969, -0.0025361, -0.0019690, -0.0003533, 0.0003460
4: -0.0018752, -0.0013180, -0.0019054, -0.0012914, -0.0003746, 0.0003825
5: 0.0053606, 0.0058879, 0.0053355, 0.0059165, -0.0003619, 0.0003545
6: -0.0010310, 0.0010610, -0.0011307, 0.0011748, -0.0014361, 0.0014064
7: -0.0040017, -0.0011526, -0.0041566, -0.0010168, -0.0019154, 0.0019558
8: 0.9863949, 0.9884019, 0.9862859, 0.9884976, -0.0013492, 0.0013777
9: -0.0053594, -0.0035375, -0.0054462, -0.0034385, -0.0012506, 0.0012248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006253
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006269
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0037420, 0.0047347, -0.0005006, 0.0005879
1: 0.0018822, 0.0019951, 0.0018629, 0.0020063, -0.0000723, 0.0000849
2: 0.0117850, 0.0122170, 0.0117422, 0.0122910, -0.0003250, 0.0002767
3: -0.0024918, -0.0020450, -0.0025362, -0.0019685, -0.0003362, 0.0002862
4: -0.0018231, -0.0013394, -0.0019059, -0.0012914, -0.0003099, 0.0003639
5: 0.0053809, 0.0058386, 0.0053355, 0.0059170, -0.0003444, 0.0002932
6: -0.0009506, 0.0008656, -0.0011308, 0.0011766, -0.0013664, 0.0011634
7: -0.0037356, -0.0012621, -0.0041591, -0.0010166, -0.0015845, 0.0018609
8: 0.9865823, 0.9883248, 0.9862842, 0.9884977, -0.0011161, 0.0013109
9: -0.0052894, -0.0037077, -0.0054463, -0.0034369, -0.0011899, 0.0010132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006253
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006253
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037879, 0.0047060, 0.0037420, 0.0047347, -0.0006130, 0.0006556
1: 0.0018695, 0.0020022, 0.0018629, 0.0020063, -0.0000886, 0.0000947
2: 0.0117580, 0.0122656, 0.0117422, 0.0122910, -0.0003625, 0.0003389
3: -0.0025197, -0.0019948, -0.0025362, -0.0019685, -0.0003749, 0.0003505
4: -0.0018775, -0.0013092, -0.0019059, -0.0012914, -0.0003795, 0.0004058
5: 0.0053523, 0.0058901, 0.0053355, 0.0059170, -0.0003841, 0.0003591
6: -0.0010640, 0.0010699, -0.0011308, 0.0011766, -0.0015238, 0.0014248
7: -0.0040138, -0.0011076, -0.0041591, -0.0010166, -0.0019405, 0.0020753
8: 0.9863865, 0.9884337, 0.9862842, 0.9884977, -0.0013669, 0.0014619
9: -0.0053881, -0.0035298, -0.0054463, -0.0034369, -0.0013270, 0.0012408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006253
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006253
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0037664, 0.0046925, -0.0004580, 0.0005248
1: 0.0018830, 0.0019929, 0.0018664, 0.0020002, -0.0000662, 0.0000758
2: 0.0117937, 0.0122142, 0.0117655, 0.0122775, -0.0002901, 0.0002532
3: -0.0024829, -0.0020480, -0.0025120, -0.0019825, -0.0003001, 0.0002619
4: -0.0018199, -0.0013491, -0.0018908, -0.0013175, -0.0002835, 0.0003248
5: 0.0053901, 0.0058356, 0.0053602, 0.0059027, -0.0003074, 0.0002683
6: -0.0009141, 0.0008537, -0.0010327, 0.0011198, -0.0012197, 0.0010646
7: -0.0037193, -0.0013118, -0.0040817, -0.0011503, -0.0014498, 0.0016612
8: 0.9865939, 0.9882898, 0.9863386, 0.9884036, -0.0010213, 0.0011702
9: -0.0052575, -0.0037181, -0.0053608, -0.0034864, -0.0010622, 0.0009271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005404, upper bound: 0.0006061
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005404, upper bound: 0.0006163
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0036562, 0.0047810, -0.0005540, 0.0006372
1: 0.0018830, 0.0019929, 0.0018505, 0.0020130, -0.0000800, 0.0000921
2: 0.0117937, 0.0122142, 0.0117166, 0.0123384, -0.0003523, 0.0003063
3: -0.0024829, -0.0020480, -0.0025626, -0.0019195, -0.0003643, 0.0003168
4: -0.0018199, -0.0013491, -0.0019590, -0.0012628, -0.0003429, 0.0003944
5: 0.0053901, 0.0058356, 0.0053083, 0.0059672, -0.0003733, 0.0003245
6: -0.0009141, 0.0008537, -0.0012384, 0.0013759, -0.0014810, 0.0012877
7: -0.0037193, -0.0013118, -0.0044305, -0.0008701, -0.0017537, 0.0020170
8: 0.9865939, 0.9882898, 0.9860929, 0.9886009, -0.0012354, 0.0014208
9: -0.0052575, -0.0037181, -0.0055400, -0.0032633, -0.0012897, 0.0011214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006950
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0007058
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037917, 0.0046918, 0.0036562, 0.0047810, -0.0005178, 0.0005657
1: 0.0018701, 0.0020001, 0.0018505, 0.0020130, -0.0000748, 0.0000817
2: 0.0117659, 0.0122635, 0.0117166, 0.0123384, -0.0003127, 0.0002863
3: -0.0025116, -0.0019969, -0.0025626, -0.0019195, -0.0003235, 0.0002961
4: -0.0018752, -0.0013180, -0.0019590, -0.0012628, -0.0003205, 0.0003502
5: 0.0053606, 0.0058879, 0.0053083, 0.0059672, -0.0003314, 0.0003033
6: -0.0010310, 0.0010610, -0.0012384, 0.0013759, -0.0013148, 0.0012034
7: -0.0040017, -0.0011526, -0.0044305, -0.0008701, -0.0016389, 0.0017906
8: 0.9863949, 0.9884019, 0.9860929, 0.9886009, -0.0011545, 0.0012613
9: -0.0053594, -0.0035375, -0.0055400, -0.0032633, -0.0011450, 0.0010480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006253
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006269
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0037664, 0.0046926, -0.0004666, 0.0005848
1: 0.0018822, 0.0019951, 0.0018664, 0.0020002, -0.0000674, 0.0000845
2: 0.0117850, 0.0122170, 0.0117655, 0.0122775, -0.0003233, 0.0002580
3: -0.0024918, -0.0020450, -0.0025121, -0.0019825, -0.0003344, 0.0002668
4: -0.0018231, -0.0013394, -0.0018908, -0.0013175, -0.0002889, 0.0003620
5: 0.0053809, 0.0058386, 0.0053601, 0.0059027, -0.0003426, 0.0002733
6: -0.0009506, 0.0008656, -0.0010328, 0.0011198, -0.0013592, 0.0010846
7: -0.0037356, -0.0012621, -0.0040818, -0.0011501, -0.0014771, 0.0018511
8: 0.9865823, 0.9883248, 0.9863386, 0.9884037, -0.0010405, 0.0013040
9: -0.0052894, -0.0037077, -0.0053610, -0.0034863, -0.0011836, 0.0009445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005437, upper bound: 0.0006062
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005437, upper bound: 0.0006061
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0036550, 0.0047811, -0.0005628, 0.0006844
1: 0.0018822, 0.0019951, 0.0018503, 0.0020130, -0.0000813, 0.0000989
2: 0.0117850, 0.0122170, 0.0117165, 0.0123391, -0.0003784, 0.0003112
3: -0.0024918, -0.0020450, -0.0025627, -0.0019188, -0.0003913, 0.0003218
4: -0.0018231, -0.0013394, -0.0019598, -0.0012627, -0.0003484, 0.0004237
5: 0.0053809, 0.0058386, 0.0053083, 0.0059680, -0.0004009, 0.0003297
6: -0.0009506, 0.0008656, -0.0012385, 0.0013788, -0.0015907, 0.0013081
7: -0.0037356, -0.0012621, -0.0044345, -0.0008699, -0.0017816, 0.0021664
8: 0.9865823, 0.9883248, 0.9860901, 0.9886011, -0.0012550, 0.0015261
9: -0.0052894, -0.0037077, -0.0055401, -0.0032608, -0.0013853, 0.0011392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006950
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006950
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037879, 0.0047060, 0.0036550, 0.0047811, -0.0005266, 0.0006132
1: 0.0018695, 0.0020022, 0.0018503, 0.0020130, -0.0000761, 0.0000886
2: 0.0117580, 0.0122656, 0.0117165, 0.0123391, -0.0003390, 0.0002912
3: -0.0025197, -0.0019948, -0.0025627, -0.0019188, -0.0003506, 0.0003011
4: -0.0018775, -0.0013092, -0.0019598, -0.0012627, -0.0003260, 0.0003796
5: 0.0053523, 0.0058901, 0.0053083, 0.0059680, -0.0003592, 0.0003085
6: -0.0010640, 0.0010699, -0.0012385, 0.0013788, -0.0014252, 0.0012240
7: -0.0040138, -0.0011076, -0.0044345, -0.0008699, -0.0016670, 0.0019410
8: 0.9863865, 0.9884337, 0.9860901, 0.9886011, -0.0011743, 0.0013673
9: -0.0053881, -0.0035298, -0.0055401, -0.0032608, -0.0012411, 0.0010659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006253
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006253
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0038553, 0.0046422, -0.0005150, 0.0005258
1: 0.0018667, 0.0020062, 0.0018793, 0.0019930, -0.0000744, 0.0000760
2: 0.0117426, 0.0122764, 0.0117933, 0.0122283, -0.0002907, 0.0002847
3: -0.0025357, -0.0019836, -0.0024832, -0.0020333, -0.0003007, 0.0002945
4: -0.0018895, -0.0012919, -0.0018358, -0.0013487, -0.0003188, 0.0003255
5: 0.0053359, 0.0059015, 0.0053897, 0.0058506, -0.0003080, 0.0003017
6: -0.0011290, 0.0011151, -0.0009157, 0.0009131, -0.0012221, 0.0011970
7: -0.0040754, -0.0010191, -0.0038003, -0.0013096, -0.0016303, 0.0016644
8: 0.9863431, 0.9884960, 0.9865368, 0.9882914, -0.0011484, 0.0011725
9: -0.0054447, -0.0034904, -0.0052589, -0.0036663, -0.0010643, 0.0010424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005163
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0038553, 0.0046422, -0.0006142, 0.0005878
1: 0.0018544, 0.0020129, 0.0018793, 0.0019930, -0.0000887, 0.0000849
2: 0.0117170, 0.0123237, 0.0117933, 0.0122283, -0.0003250, 0.0003396
3: -0.0025622, -0.0019347, -0.0024832, -0.0020333, -0.0003361, 0.0003512
4: -0.0019425, -0.0012632, -0.0018358, -0.0013487, -0.0003802, 0.0003639
5: 0.0053088, 0.0059516, 0.0053897, 0.0058506, -0.0003444, 0.0003598
6: -0.0012366, 0.0013139, -0.0009157, 0.0009131, -0.0013663, 0.0014277
7: -0.0043461, -0.0008726, -0.0038003, -0.0013096, -0.0019444, 0.0018608
8: 0.9861523, 0.9885992, 0.9865368, 0.9882914, -0.0013697, 0.0013108
9: -0.0055384, -0.0033173, -0.0052589, -0.0036663, -0.0011898, 0.0012433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005163
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0037428, 0.0047347, -0.0004241, 0.0004565
1: 0.0018667, 0.0020062, 0.0018630, 0.0020063, -0.0000613, 0.0000660
2: 0.0117426, 0.0122764, 0.0117422, 0.0122906, -0.0002524, 0.0002345
3: -0.0025357, -0.0019836, -0.0025361, -0.0019690, -0.0002610, 0.0002425
4: -0.0018895, -0.0012919, -0.0019054, -0.0012914, -0.0002626, 0.0002826
5: 0.0053359, 0.0059015, 0.0053355, 0.0059165, -0.0002674, 0.0002485
6: -0.0011290, 0.0011151, -0.0011307, 0.0011748, -0.0010611, 0.0009858
7: -0.0040754, -0.0010191, -0.0041566, -0.0010168, -0.0013426, 0.0014451
8: 0.9863431, 0.9884960, 0.9862859, 0.9884976, -0.0009458, 0.0010179
9: -0.0054447, -0.0034904, -0.0054462, -0.0034385, -0.0009240, 0.0008585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005163
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0037428, 0.0047347, -0.0005383, 0.0005380
1: 0.0018544, 0.0020129, 0.0018630, 0.0020063, -0.0000778, 0.0000777
2: 0.0117170, 0.0123237, 0.0117422, 0.0122906, -0.0002974, 0.0002976
3: -0.0025622, -0.0019347, -0.0025361, -0.0019690, -0.0003076, 0.0003078
4: -0.0019425, -0.0012632, -0.0019054, -0.0012914, -0.0003332, 0.0003330
5: 0.0053088, 0.0059516, 0.0053355, 0.0059165, -0.0003151, 0.0003154
6: -0.0012366, 0.0013139, -0.0011307, 0.0011748, -0.0012504, 0.0012512
7: -0.0043461, -0.0008726, -0.0041566, -0.0010168, -0.0017041, 0.0017029
8: 0.9861523, 0.9885992, 0.9862859, 0.9884976, -0.0012004, 0.0011996
9: -0.0055384, -0.0033173, -0.0054462, -0.0034385, -0.0010889, 0.0010896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005163
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0038532, 0.0046422, -0.0005240, 0.0005575
1: 0.0018663, 0.0020083, 0.0018790, 0.0019930, -0.0000757, 0.0000805
2: 0.0117347, 0.0122781, 0.0117933, 0.0122295, -0.0003082, 0.0002897
3: -0.0025439, -0.0019819, -0.0024833, -0.0020321, -0.0003188, 0.0002996
4: -0.0018914, -0.0012831, -0.0018371, -0.0013487, -0.0003243, 0.0003451
5: 0.0053276, 0.0059033, 0.0053896, 0.0058518, -0.0003266, 0.0003069
6: -0.0011620, 0.0011222, -0.0009158, 0.0009181, -0.0012957, 0.0012178
7: -0.0040850, -0.0009741, -0.0038071, -0.0013094, -0.0016586, 0.0017647
8: 0.9863362, 0.9885277, 0.9865320, 0.9882915, -0.0011683, 0.0012431
9: -0.0054735, -0.0034843, -0.0052591, -0.0036620, -0.0011284, 0.0010605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0038532, 0.0046422, -0.0006233, 0.0006149
1: 0.0018536, 0.0020156, 0.0018790, 0.0019930, -0.0000901, 0.0000888
2: 0.0117068, 0.0123267, 0.0117933, 0.0122295, -0.0003400, 0.0003446
3: -0.0025727, -0.0019316, -0.0024833, -0.0020321, -0.0003516, 0.0003564
4: -0.0019459, -0.0012518, -0.0018371, -0.0013487, -0.0003858, 0.0003806
5: 0.0052980, 0.0059548, 0.0053896, 0.0058518, -0.0003602, 0.0003651
6: -0.0012793, 0.0013267, -0.0009158, 0.0009181, -0.0014292, 0.0014488
7: -0.0043636, -0.0008144, -0.0038071, -0.0013094, -0.0019731, 0.0019465
8: 0.9861401, 0.9886402, 0.9865320, 0.9882915, -0.0013899, 0.0013712
9: -0.0055756, -0.0033061, -0.0052591, -0.0036620, -0.0012446, 0.0012616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0037420, 0.0047347, -0.0004301, 0.0005177
1: 0.0018663, 0.0020083, 0.0018629, 0.0020063, -0.0000621, 0.0000748
2: 0.0117347, 0.0122781, 0.0117422, 0.0122910, -0.0002862, 0.0002378
3: -0.0025439, -0.0019819, -0.0025362, -0.0019685, -0.0002961, 0.0002459
4: -0.0018914, -0.0012831, -0.0019059, -0.0012914, -0.0002662, 0.0003205
5: 0.0053276, 0.0059033, 0.0053355, 0.0059170, -0.0003033, 0.0002519
6: -0.0011620, 0.0011222, -0.0011308, 0.0011766, -0.0012034, 0.0009996
7: -0.0040850, -0.0009741, -0.0041591, -0.0010166, -0.0013613, 0.0016389
8: 0.9863362, 0.9885277, 0.9862842, 0.9884977, -0.0009590, 0.0011545
9: -0.0054735, -0.0034843, -0.0054463, -0.0034369, -0.0010480, 0.0008705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0037420, 0.0047347, -0.0005443, 0.0005890
1: 0.0018536, 0.0020156, 0.0018629, 0.0020063, -0.0000786, 0.0000851
2: 0.0117068, 0.0123267, 0.0117422, 0.0122910, -0.0003256, 0.0003009
3: -0.0025727, -0.0019316, -0.0025362, -0.0019685, -0.0003368, 0.0003112
4: -0.0019459, -0.0012518, -0.0019059, -0.0012914, -0.0003369, 0.0003646
5: 0.0052980, 0.0059548, 0.0053355, 0.0059170, -0.0003450, 0.0003189
6: -0.0012793, 0.0013267, -0.0011308, 0.0011766, -0.0013690, 0.0012651
7: -0.0043636, -0.0008144, -0.0041591, -0.0010166, -0.0017230, 0.0018644
8: 0.9861401, 0.9886402, 0.9862842, 0.9884977, -0.0012137, 0.0013133
9: -0.0055756, -0.0033061, -0.0054463, -0.0034369, -0.0011922, 0.0011017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0037664, 0.0046925, -0.0005934, 0.0006371
1: 0.0018667, 0.0020062, 0.0018664, 0.0020002, -0.0000857, 0.0000920
2: 0.0117426, 0.0122764, 0.0117655, 0.0122775, -0.0003522, 0.0003281
3: -0.0025357, -0.0019836, -0.0025120, -0.0019825, -0.0003643, 0.0003393
4: -0.0018895, -0.0012919, -0.0018908, -0.0013175, -0.0003673, 0.0003944
5: 0.0053359, 0.0059015, 0.0053602, 0.0059027, -0.0003732, 0.0003476
6: -0.0011290, 0.0011151, -0.0010327, 0.0011198, -0.0014808, 0.0013793
7: -0.0040754, -0.0010191, -0.0040817, -0.0011503, -0.0018784, 0.0020167
8: 0.9863431, 0.9884960, 0.9863386, 0.9884036, -0.0013232, 0.0014206
9: -0.0054447, -0.0034904, -0.0053608, -0.0034864, -0.0012895, 0.0012011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005797
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005948
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0037664, 0.0046925, -0.0005413, 0.0005516
1: 0.0018544, 0.0020129, 0.0018664, 0.0020002, -0.0000782, 0.0000797
2: 0.0117170, 0.0123237, 0.0117655, 0.0122775, -0.0003050, 0.0002993
3: -0.0025622, -0.0019347, -0.0025120, -0.0019825, -0.0003154, 0.0003095
4: -0.0019425, -0.0012632, -0.0018908, -0.0013175, -0.0003351, 0.0003415
5: 0.0053088, 0.0059516, 0.0053602, 0.0059027, -0.0003231, 0.0003171
6: -0.0012366, 0.0013139, -0.0010327, 0.0011198, -0.0012821, 0.0012582
7: -0.0043461, -0.0008726, -0.0040817, -0.0011503, -0.0017135, 0.0017461
8: 0.9861523, 0.9885992, 0.9863386, 0.9884036, -0.0012071, 0.0012300
9: -0.0055384, -0.0033173, -0.0053608, -0.0034864, -0.0011165, 0.0010957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005163
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005253
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0036562, 0.0047810, -0.0005055, 0.0005701
1: 0.0018667, 0.0020062, 0.0018505, 0.0020130, -0.0000730, 0.0000824
2: 0.0117426, 0.0122764, 0.0117166, 0.0123384, -0.0003152, 0.0002795
3: -0.0025357, -0.0019836, -0.0025626, -0.0019195, -0.0003260, 0.0002891
4: -0.0018895, -0.0012919, -0.0019590, -0.0012628, -0.0003129, 0.0003529
5: 0.0053359, 0.0059015, 0.0053083, 0.0059672, -0.0003340, 0.0002961
6: -0.0011290, 0.0011151, -0.0012384, 0.0013759, -0.0013251, 0.0011749
7: -0.0040754, -0.0010191, -0.0044305, -0.0008701, -0.0016002, 0.0018047
8: 0.9863431, 0.9884960, 0.9860929, 0.9886009, -0.0011272, 0.0012713
9: -0.0054447, -0.0034904, -0.0055400, -0.0032633, -0.0011540, 0.0010232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005797
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005948
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0036562, 0.0047810, -0.0004511, 0.0004835
1: 0.0018544, 0.0020129, 0.0018505, 0.0020130, -0.0000652, 0.0000699
2: 0.0117170, 0.0123237, 0.0117166, 0.0123384, -0.0002673, 0.0002494
3: -0.0025622, -0.0019347, -0.0025626, -0.0019195, -0.0002765, 0.0002580
4: -0.0019425, -0.0012632, -0.0019590, -0.0012628, -0.0002793, 0.0002993
5: 0.0053088, 0.0059516, 0.0053083, 0.0059672, -0.0002832, 0.0002643
6: -0.0012366, 0.0013139, -0.0012384, 0.0013759, -0.0011238, 0.0010486
7: -0.0043461, -0.0008726, -0.0044305, -0.0008701, -0.0014281, 0.0015306
8: 0.9861523, 0.9885992, 0.9860929, 0.9886009, -0.0010060, 0.0010782
9: -0.0055384, -0.0033173, -0.0055400, -0.0032633, -0.0009787, 0.0009131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005163
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005253
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0037664, 0.0046926, -0.0006023, 0.0006644
1: 0.0018663, 0.0020083, 0.0018664, 0.0020002, -0.0000870, 0.0000960
2: 0.0117347, 0.0122781, 0.0117655, 0.0122775, -0.0003673, 0.0003330
3: -0.0025439, -0.0019819, -0.0025121, -0.0019825, -0.0003799, 0.0003444
4: -0.0018914, -0.0012831, -0.0018908, -0.0013175, -0.0003728, 0.0004113
5: 0.0053276, 0.0059033, 0.0053601, 0.0059027, -0.0003892, 0.0003528
6: -0.0011620, 0.0011222, -0.0010328, 0.0011198, -0.0015443, 0.0014000
7: -0.0040850, -0.0009741, -0.0040818, -0.0011501, -0.0019066, 0.0021032
8: 0.9863362, 0.9885277, 0.9863386, 0.9884037, -0.0013431, 0.0014816
9: -0.0054735, -0.0034843, -0.0053610, -0.0034863, -0.0013449, 0.0012191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005797
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005797
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0037664, 0.0046926, -0.0005505, 0.0005821
1: 0.0018536, 0.0020156, 0.0018664, 0.0020002, -0.0000795, 0.0000841
2: 0.0117068, 0.0123267, 0.0117655, 0.0122775, -0.0003218, 0.0003044
3: -0.0025727, -0.0019316, -0.0025121, -0.0019825, -0.0003329, 0.0003148
4: -0.0019459, -0.0012518, -0.0018908, -0.0013175, -0.0003408, 0.0003603
5: 0.0052980, 0.0059548, 0.0053601, 0.0059027, -0.0003410, 0.0003225
6: -0.0012793, 0.0013267, -0.0010328, 0.0011198, -0.0013530, 0.0012796
7: -0.0043636, -0.0008144, -0.0040818, -0.0011501, -0.0017427, 0.0018427
8: 0.9861401, 0.9886402, 0.9863386, 0.9884037, -0.0012276, 0.0012980
9: -0.0055756, -0.0033061, -0.0053610, -0.0034863, -0.0011783, 0.0011143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005163
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005163
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0036550, 0.0047811, -0.0005113, 0.0006283
1: 0.0018663, 0.0020083, 0.0018503, 0.0020130, -0.0000739, 0.0000908
2: 0.0117347, 0.0122781, 0.0117165, 0.0123391, -0.0003474, 0.0002827
3: -0.0025439, -0.0019819, -0.0025627, -0.0019188, -0.0003593, 0.0002924
4: -0.0018914, -0.0012831, -0.0019598, -0.0012627, -0.0003165, 0.0003890
5: 0.0053276, 0.0059033, 0.0053083, 0.0059680, -0.0003681, 0.0002995
6: -0.0011620, 0.0011222, -0.0012385, 0.0013788, -0.0014604, 0.0011884
7: -0.0040850, -0.0009741, -0.0044345, -0.0008699, -0.0016184, 0.0019890
8: 0.9863362, 0.9885277, 0.9860901, 0.9886011, -0.0011401, 0.0014011
9: -0.0054735, -0.0034843, -0.0055401, -0.0032608, -0.0012718, 0.0010349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005797
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005797
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0036550, 0.0047811, -0.0004590, 0.0005435
1: 0.0018536, 0.0020156, 0.0018503, 0.0020130, -0.0000663, 0.0000785
2: 0.0117068, 0.0123267, 0.0117165, 0.0123391, -0.0003005, 0.0002538
3: -0.0025727, -0.0019316, -0.0025627, -0.0019188, -0.0003108, 0.0002625
4: -0.0019459, -0.0012518, -0.0019598, -0.0012627, -0.0002841, 0.0003365
5: 0.0052980, 0.0059548, 0.0053083, 0.0059680, -0.0003184, 0.0002689
6: -0.0012793, 0.0013267, -0.0012385, 0.0013788, -0.0012633, 0.0010669
7: -0.0043636, -0.0008144, -0.0044345, -0.0008699, -0.0014530, 0.0017205
8: 0.9861401, 0.9886402, 0.9860901, 0.9886011, -0.0010235, 0.0012120
9: -0.0055756, -0.0033061, -0.0055401, -0.0032608, -0.0011001, 0.0009291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005163
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005163
time: 0.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006253
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006269
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006253
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005322, upper bound: 0.0006269
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006253
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006253
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006253
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005474, upper bound: 0.0006253
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005404, upper bound: 0.0006061
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005404, upper bound: 0.0006163
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006950
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0007058
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006253
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005187, upper bound: 0.0006269
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005437, upper bound: 0.0006062
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005437, upper bound: 0.0006061
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006950
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006950
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006253
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0005253, upper bound: 0.0006253
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005163
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005163
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005163
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005163
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006424, upper bound: 0.0005253
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006471, upper bound: 0.0005163
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005797
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005948
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005163
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005253
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005797
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005948
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005163
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006289, upper bound: 0.0005253
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005797
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005797
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005163
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005163
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005797
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005797
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005163
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 8, lower bound: -0.0006269, upper bound: 0.0005163

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0037684, 0.0047340, -0.0004903, 0.0005134
1: 0.0018830, 0.0019929, 0.0018667, 0.0020062, -0.0000708, 0.0000742
2: 0.0117937, 0.0122142, 0.0117426, 0.0122764, -0.0002838, 0.0002711
3: -0.0024829, -0.0020480, -0.0025357, -0.0019836, -0.0002935, 0.0002804
4: -0.0018199, -0.0013491, -0.0018895, -0.0012919, -0.0003035, 0.0003178
5: 0.0053901, 0.0058356, 0.0053359, 0.0059015, -0.0003007, 0.0002872
6: -0.0009141, 0.0008537, -0.0011290, 0.0011151, -0.0011932, 0.0011396
7: -0.0037193, -0.0013118, -0.0040754, -0.0010191, -0.0015520, 0.0016250
8: 0.9865939, 0.9882898, 0.9863431, 0.9884960, -0.0010933, 0.0011447
9: -0.0052575, -0.0037181, -0.0054447, -0.0034904, -0.0010391, 0.0009924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005082, upper bound: 0.0006176
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005111, upper bound: 0.0006176
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0037654, 0.0047482, -0.0005064, 0.0005163
1: 0.0018830, 0.0019929, 0.0018663, 0.0020083, -0.0000732, 0.0000746
2: 0.0117937, 0.0122142, 0.0117347, 0.0122781, -0.0002855, 0.0002800
3: -0.0024829, -0.0020480, -0.0025439, -0.0019819, -0.0002952, 0.0002896
4: -0.0018199, -0.0013491, -0.0018914, -0.0012831, -0.0003135, 0.0003196
5: 0.0053901, 0.0058356, 0.0053276, 0.0059033, -0.0003025, 0.0002967
6: -0.0009141, 0.0008537, -0.0011620, 0.0011222, -0.0012001, 0.0011770
7: -0.0037193, -0.0013118, -0.0040850, -0.0009741, -0.0016030, 0.0016345
8: 0.9865939, 0.9882898, 0.9863362, 0.9885277, -0.0011292, 0.0011513
9: -0.0052575, -0.0037181, -0.0054735, -0.0034843, -0.0010451, 0.0010250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005082, upper bound: 0.0006238
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005111, upper bound: 0.0006238
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037917, 0.0046918, 0.0037684, 0.0047340, -0.0006035, 0.0005918
1: 0.0018701, 0.0020001, 0.0018667, 0.0020062, -0.0000872, 0.0000855
2: 0.0117659, 0.0122635, 0.0117426, 0.0122764, -0.0003272, 0.0003337
3: -0.0025116, -0.0019969, -0.0025357, -0.0019836, -0.0003384, 0.0003451
4: -0.0018752, -0.0013180, -0.0018895, -0.0012919, -0.0003736, 0.0003663
5: 0.0053606, 0.0058879, 0.0053359, 0.0059015, -0.0003467, 0.0003535
6: -0.0010310, 0.0010610, -0.0011290, 0.0011151, -0.0013755, 0.0014028
7: -0.0040017, -0.0011526, -0.0040754, -0.0010191, -0.0019105, 0.0018733
8: 0.9863949, 0.9884019, 0.9863431, 0.9884960, -0.0013458, 0.0013196
9: -0.0053594, -0.0035375, -0.0054447, -0.0034904, -0.0011979, 0.0012216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005542, upper bound: 0.0006010
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005586, upper bound: 0.0006012
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037917, 0.0046918, 0.0037654, 0.0047482, -0.0006197, 0.0005948
1: 0.0018701, 0.0020001, 0.0018663, 0.0020083, -0.0000895, 0.0000859
2: 0.0117659, 0.0122635, 0.0117347, 0.0122781, -0.0003288, 0.0003426
3: -0.0025116, -0.0019969, -0.0025439, -0.0019819, -0.0003401, 0.0003543
4: -0.0018752, -0.0013180, -0.0018914, -0.0012831, -0.0003836, 0.0003682
5: 0.0053606, 0.0058879, 0.0053276, 0.0059033, -0.0003484, 0.0003630
6: -0.0010310, 0.0010610, -0.0011620, 0.0011222, -0.0013824, 0.0014402
7: -0.0040017, -0.0011526, -0.0040850, -0.0009741, -0.0019615, 0.0018828
8: 0.9863949, 0.9884019, 0.9863362, 0.9885277, -0.0013817, 0.0013262
9: -0.0053594, -0.0035375, -0.0054735, -0.0034843, -0.0012039, 0.0012542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005542, upper bound: 0.0006019
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005586, upper bound: 0.0006024
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0037684, 0.0047340, -0.0005280, 0.0005622
1: 0.0018822, 0.0019951, 0.0018667, 0.0020062, -0.0000763, 0.0000812
2: 0.0117850, 0.0122170, 0.0117426, 0.0122764, -0.0003108, 0.0002919
3: -0.0024918, -0.0020450, -0.0025357, -0.0019836, -0.0003214, 0.0003019
4: -0.0018231, -0.0013394, -0.0018895, -0.0012919, -0.0003268, 0.0003480
5: 0.0053809, 0.0058386, 0.0053359, 0.0059015, -0.0003293, 0.0003093
6: -0.0009506, 0.0008656, -0.0011290, 0.0011151, -0.0013066, 0.0012272
7: -0.0037356, -0.0012621, -0.0040754, -0.0010191, -0.0016713, 0.0017795
8: 0.9865823, 0.9883248, 0.9863431, 0.9884960, -0.0011773, 0.0012535
9: -0.0052894, -0.0037077, -0.0054447, -0.0034904, -0.0011379, 0.0010687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005245, upper bound: 0.0006176
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005246, upper bound: 0.0006176
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0037654, 0.0047482, -0.0004995, 0.0005226
1: 0.0018822, 0.0019951, 0.0018663, 0.0020083, -0.0000722, 0.0000755
2: 0.0117850, 0.0122170, 0.0117347, 0.0122781, -0.0002889, 0.0002762
3: -0.0024918, -0.0020450, -0.0025439, -0.0019819, -0.0002988, 0.0002856
4: -0.0018231, -0.0013394, -0.0018914, -0.0012831, -0.0003092, 0.0003235
5: 0.0053809, 0.0058386, 0.0053276, 0.0059033, -0.0003061, 0.0002926
6: -0.0009506, 0.0008656, -0.0011620, 0.0011222, -0.0012146, 0.0011611
7: -0.0037356, -0.0012621, -0.0040850, -0.0009741, -0.0015813, 0.0016541
8: 0.9865823, 0.9883248, 0.9863362, 0.9885277, -0.0011139, 0.0011652
9: -0.0052894, -0.0037077, -0.0054735, -0.0034843, -0.0010577, 0.0010111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005245, upper bound: 0.0006176
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005246, upper bound: 0.0006176
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037879, 0.0047060, 0.0037684, 0.0047340, -0.0006301, 0.0006299
1: 0.0018695, 0.0020022, 0.0018667, 0.0020062, -0.0000910, 0.0000910
2: 0.0117580, 0.0122656, 0.0117426, 0.0122764, -0.0003482, 0.0003484
3: -0.0025197, -0.0019948, -0.0025357, -0.0019836, -0.0003602, 0.0003603
4: -0.0018775, -0.0013092, -0.0018895, -0.0012919, -0.0003900, 0.0003899
5: 0.0053523, 0.0058901, 0.0053359, 0.0059015, -0.0003690, 0.0003691
6: -0.0010640, 0.0010699, -0.0011290, 0.0011151, -0.0014640, 0.0014645
7: -0.0040138, -0.0011076, -0.0040754, -0.0010191, -0.0019945, 0.0019939
8: 0.9863865, 0.9884337, 0.9863431, 0.9884960, -0.0014050, 0.0014045
9: -0.0053881, -0.0035298, -0.0054447, -0.0034904, -0.0012749, 0.0012754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005719, upper bound: 0.0006010
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005734, upper bound: 0.0006012
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037879, 0.0047060, 0.0037654, 0.0047482, -0.0006120, 0.0006011
1: 0.0018695, 0.0020022, 0.0018663, 0.0020083, -0.0000884, 0.0000868
2: 0.0117580, 0.0122656, 0.0117347, 0.0122781, -0.0003323, 0.0003384
3: -0.0025197, -0.0019948, -0.0025439, -0.0019819, -0.0003437, 0.0003500
4: -0.0018775, -0.0013092, -0.0018914, -0.0012831, -0.0003788, 0.0003721
5: 0.0053523, 0.0058901, 0.0053276, 0.0059033, -0.0003521, 0.0003585
6: -0.0010640, 0.0010699, -0.0011620, 0.0011222, -0.0013971, 0.0014225
7: -0.0040138, -0.0011076, -0.0040850, -0.0009741, -0.0019373, 0.0019027
8: 0.9863865, 0.9884337, 0.9863362, 0.9885277, -0.0013647, 0.0013403
9: -0.0053881, -0.0035298, -0.0054735, -0.0034843, -0.0012167, 0.0012388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005719, upper bound: 0.0006010
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005734, upper bound: 0.0006012
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0037917, 0.0046918, -0.0004564, 0.0004912
1: 0.0018830, 0.0019929, 0.0018701, 0.0020001, -0.0000659, 0.0000710
2: 0.0117937, 0.0122142, 0.0117659, 0.0122635, -0.0002716, 0.0002523
3: -0.0024829, -0.0020480, -0.0025116, -0.0019969, -0.0002809, 0.0002610
4: -0.0018199, -0.0013491, -0.0018752, -0.0013180, -0.0002825, 0.0003041
5: 0.0053901, 0.0058356, 0.0053606, 0.0058879, -0.0002878, 0.0002674
6: -0.0009141, 0.0008537, -0.0010310, 0.0010610, -0.0011417, 0.0010608
7: -0.0037193, -0.0013118, -0.0040017, -0.0011526, -0.0014447, 0.0015549
8: 0.9865939, 0.9882898, 0.9863949, 0.9884019, -0.0010177, 0.0010953
9: -0.0052575, -0.0037181, -0.0053594, -0.0035375, -0.0009943, 0.0009238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005142, upper bound: 0.0005836
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005191, upper bound: 0.0005836
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0037879, 0.0047060, -0.0004945, 0.0005178
1: 0.0018830, 0.0019929, 0.0018695, 0.0020022, -0.0000714, 0.0000748
2: 0.0117937, 0.0122142, 0.0117580, 0.0122656, -0.0002863, 0.0002734
3: -0.0024829, -0.0020480, -0.0025197, -0.0019948, -0.0002961, 0.0002827
4: -0.0018199, -0.0013491, -0.0018775, -0.0013092, -0.0003061, 0.0003205
5: 0.0053901, 0.0058356, 0.0053523, 0.0058901, -0.0003033, 0.0002897
6: -0.0009141, 0.0008537, -0.0010640, 0.0010699, -0.0012034, 0.0011493
7: -0.0037193, -0.0013118, -0.0040138, -0.0011076, -0.0015653, 0.0016390
8: 0.9865939, 0.9882898, 0.9863865, 0.9884337, -0.0011026, 0.0011545
9: -0.0052575, -0.0037181, -0.0053881, -0.0035298, -0.0010480, 0.0010009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005142, upper bound: 0.0005934
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005191, upper bound: 0.0005934
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0036829, 0.0047802, -0.0005523, 0.0006126
1: 0.0018830, 0.0019929, 0.0018544, 0.0020129, -0.0000798, 0.0000885
2: 0.0117937, 0.0122142, 0.0117170, 0.0123237, -0.0003387, 0.0003054
3: -0.0024829, -0.0020480, -0.0025622, -0.0019347, -0.0003503, 0.0003158
4: -0.0018199, -0.0013491, -0.0019425, -0.0012632, -0.0003419, 0.0003792
5: 0.0053901, 0.0058356, 0.0053088, 0.0059516, -0.0003589, 0.0003235
6: -0.0009141, 0.0008537, -0.0012366, 0.0013139, -0.0014238, 0.0012837
7: -0.0037193, -0.0013118, -0.0043461, -0.0008726, -0.0017484, 0.0019392
8: 0.9865939, 0.9882898, 0.9861523, 0.9885992, -0.0012316, 0.0013660
9: -0.0052575, -0.0037181, -0.0055384, -0.0033173, -0.0012399, 0.0011179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004955, upper bound: 0.0006730
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004981, upper bound: 0.0006730
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0038809, 0.0046415, 0.0036774, 0.0047986, -0.0005638, 0.0006103
1: 0.0018830, 0.0019929, 0.0018536, 0.0020156, -0.0000815, 0.0000882
2: 0.0117937, 0.0122142, 0.0117068, 0.0123267, -0.0003374, 0.0003117
3: -0.0024829, -0.0020480, -0.0025727, -0.0019316, -0.0003490, 0.0003224
4: -0.0018199, -0.0013491, -0.0019459, -0.0012518, -0.0003490, 0.0003778
5: 0.0053901, 0.0058356, 0.0052980, 0.0059548, -0.0003575, 0.0003303
6: -0.0009141, 0.0008537, -0.0012793, 0.0013267, -0.0014185, 0.0013105
7: -0.0037193, -0.0013118, -0.0043636, -0.0008144, -0.0017848, 0.0019319
8: 0.9865939, 0.9882898, 0.9861401, 0.9886402, -0.0012573, 0.0013608
9: -0.0052575, -0.0037181, -0.0055756, -0.0033061, -0.0012353, 0.0011413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004955, upper bound: 0.0006818
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004981, upper bound: 0.0006820
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037917, 0.0046918, 0.0036829, 0.0047802, -0.0005162, 0.0005397
1: 0.0018701, 0.0020001, 0.0018544, 0.0020129, -0.0000746, 0.0000780
2: 0.0117659, 0.0122635, 0.0117170, 0.0123237, -0.0002984, 0.0002854
3: -0.0025116, -0.0019969, -0.0025622, -0.0019347, -0.0003086, 0.0002951
4: -0.0018752, -0.0013180, -0.0019425, -0.0012632, -0.0003195, 0.0003341
5: 0.0053606, 0.0058879, 0.0053088, 0.0059516, -0.0003161, 0.0003024
6: -0.0010310, 0.0010610, -0.0012366, 0.0013139, -0.0012544, 0.0011997
7: -0.0040017, -0.0011526, -0.0043461, -0.0008726, -0.0016339, 0.0017083
8: 0.9863949, 0.9884019, 0.9861523, 0.9885992, -0.0011510, 0.0012034
9: -0.0053594, -0.0035375, -0.0055384, -0.0033173, -0.0010923, 0.0010448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005542, upper bound: 0.0006010
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005586, upper bound: 0.0006012
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037917, 0.0046918, 0.0036774, 0.0047986, -0.0005317, 0.0005427
1: 0.0018701, 0.0020001, 0.0018536, 0.0020156, -0.0000768, 0.0000784
2: 0.0117659, 0.0122635, 0.0117068, 0.0123267, -0.0003000, 0.0002940
3: -0.0025116, -0.0019969, -0.0025727, -0.0019316, -0.0003103, 0.0003040
4: -0.0018752, -0.0013180, -0.0019459, -0.0012518, -0.0003291, 0.0003359
5: 0.0053606, 0.0058879, 0.0052980, 0.0059548, -0.0003179, 0.0003115
6: -0.0010310, 0.0010610, -0.0012793, 0.0013267, -0.0012613, 0.0012359
7: -0.0040017, -0.0011526, -0.0043636, -0.0008144, -0.0016831, 0.0017178
8: 0.9863949, 0.9884019, 0.9861401, 0.9886402, -0.0011856, 0.0012100
9: -0.0053594, -0.0035375, -0.0055756, -0.0033061, -0.0010984, 0.0010762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005542, upper bound: 0.0006019
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005586, upper bound: 0.0006024
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0037917, 0.0046918, -0.0004941, 0.0005400
1: 0.0018822, 0.0019951, 0.0018701, 0.0020001, -0.0000714, 0.0000780
2: 0.0117850, 0.0122170, 0.0117659, 0.0122635, -0.0002986, 0.0002732
3: -0.0024918, -0.0020450, -0.0025116, -0.0019969, -0.0003088, 0.0002825
4: -0.0018231, -0.0013394, -0.0018752, -0.0013180, -0.0003059, 0.0003343
5: 0.0053809, 0.0058386, 0.0053606, 0.0058879, -0.0003163, 0.0002894
6: -0.0009506, 0.0008656, -0.0010310, 0.0010610, -0.0012551, 0.0011484
7: -0.0037356, -0.0012621, -0.0040017, -0.0011526, -0.0015641, 0.0017094
8: 0.9865823, 0.9883248, 0.9863949, 0.9884019, -0.0011018, 0.0012041
9: -0.0052894, -0.0037077, -0.0053594, -0.0035375, -0.0010930, 0.0010001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005207, upper bound: 0.0005836
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005212, upper bound: 0.0005836
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0037879, 0.0047060, -0.0004654, 0.0004993
1: 0.0018822, 0.0019951, 0.0018695, 0.0020022, -0.0000672, 0.0000721
2: 0.0117850, 0.0122170, 0.0117580, 0.0122656, -0.0002761, 0.0002573
3: -0.0024918, -0.0020450, -0.0025197, -0.0019948, -0.0002855, 0.0002661
4: -0.0018231, -0.0013394, -0.0018775, -0.0013092, -0.0002881, 0.0003091
5: 0.0053809, 0.0058386, 0.0053523, 0.0058901, -0.0002925, 0.0002726
6: -0.0009506, 0.0008656, -0.0010640, 0.0010699, -0.0011606, 0.0010817
7: -0.0037356, -0.0012621, -0.0040138, -0.0011076, -0.0014732, 0.0015806
8: 0.9865823, 0.9883248, 0.9863865, 0.9884337, -0.0010378, 0.0011134
9: -0.0052894, -0.0037077, -0.0053881, -0.0035298, -0.0010107, 0.0009420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005207, upper bound: 0.0005836
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005212, upper bound: 0.0005836
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0036829, 0.0047802, -0.0005900, 0.0006614
1: 0.0018822, 0.0019951, 0.0018544, 0.0020129, -0.0000852, 0.0000956
2: 0.0117850, 0.0122170, 0.0117170, 0.0123237, -0.0003657, 0.0003262
3: -0.0024918, -0.0020450, -0.0025622, -0.0019347, -0.0003782, 0.0003374
4: -0.0018231, -0.0013394, -0.0019425, -0.0012632, -0.0003652, 0.0004094
5: 0.0053809, 0.0058386, 0.0053088, 0.0059516, -0.0003874, 0.0003456
6: -0.0009506, 0.0008656, -0.0012366, 0.0013139, -0.0015373, 0.0013714
7: -0.0037356, -0.0012621, -0.0043461, -0.0008726, -0.0018677, 0.0020936
8: 0.9865823, 0.9883248, 0.9861523, 0.9885992, -0.0013156, 0.0014748
9: -0.0052894, -0.0037077, -0.0055384, -0.0033173, -0.0013387, 0.0011942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005036, upper bound: 0.0006730
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005012, upper bound: 0.0006730
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0038758, 0.0046572, 0.0036774, 0.0047986, -0.0005617, 0.0006219
1: 0.0018822, 0.0019951, 0.0018536, 0.0020156, -0.0000811, 0.0000898
2: 0.0117850, 0.0122170, 0.0117068, 0.0123267, -0.0003438, 0.0003105
3: -0.0024918, -0.0020450, -0.0025727, -0.0019316, -0.0003556, 0.0003212
4: -0.0018231, -0.0013394, -0.0019459, -0.0012518, -0.0003477, 0.0003850
5: 0.0053809, 0.0058386, 0.0052980, 0.0059548, -0.0003643, 0.0003290
6: -0.0009506, 0.0008656, -0.0012793, 0.0013267, -0.0014455, 0.0013055
7: -0.0037356, -0.0012621, -0.0043636, -0.0008144, -0.0017779, 0.0019686
8: 0.9865823, 0.9883248, 0.9861401, 0.9886402, -0.0012524, 0.0013867
9: -0.0052894, -0.0037077, -0.0055756, -0.0033061, -0.0012588, 0.0011369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005036, upper bound: 0.0006730
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005012, upper bound: 0.0006730
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037879, 0.0047060, 0.0036829, 0.0047802, -0.0005533, 0.0005875
1: 0.0018695, 0.0020022, 0.0018544, 0.0020129, -0.0000799, 0.0000849
2: 0.0117580, 0.0122656, 0.0117170, 0.0123237, -0.0003248, 0.0003059
3: -0.0025197, -0.0019948, -0.0025622, -0.0019347, -0.0003360, 0.0003164
4: -0.0018775, -0.0013092, -0.0019425, -0.0012632, -0.0003425, 0.0003637
5: 0.0053523, 0.0058901, 0.0053088, 0.0059516, -0.0003442, 0.0003241
6: -0.0010640, 0.0010699, -0.0012366, 0.0013139, -0.0013656, 0.0012861
7: -0.0040138, -0.0011076, -0.0043461, -0.0008726, -0.0017516, 0.0018598
8: 0.9863865, 0.9884337, 0.9861523, 0.9885992, -0.0012338, 0.0013101
9: -0.0053881, -0.0035298, -0.0055384, -0.0033173, -0.0011892, 0.0011200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005719, upper bound: 0.0006010
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005734, upper bound: 0.0006012
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037879, 0.0047060, 0.0036774, 0.0047986, -0.0005256, 0.0005491
1: 0.0018695, 0.0020022, 0.0018536, 0.0020156, -0.0000759, 0.0000793
2: 0.0117580, 0.0122656, 0.0117068, 0.0123267, -0.0003036, 0.0002906
3: -0.0025197, -0.0019948, -0.0025727, -0.0019316, -0.0003140, 0.0003006
4: -0.0018775, -0.0013092, -0.0019459, -0.0012518, -0.0003254, 0.0003399
5: 0.0053523, 0.0058901, 0.0052980, 0.0059548, -0.0003217, 0.0003079
6: -0.0010640, 0.0010699, -0.0012793, 0.0013267, -0.0012764, 0.0012217
7: -0.0040138, -0.0011076, -0.0043636, -0.0008144, -0.0016639, 0.0017383
8: 0.9863865, 0.9884337, 0.9861401, 0.9886402, -0.0011721, 0.0012245
9: -0.0053881, -0.0035298, -0.0055756, -0.0033061, -0.0011115, 0.0010639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005719, upper bound: 0.0006010
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005734, upper bound: 0.0006012
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0038809, 0.0046415, -0.0005134, 0.0004903
1: 0.0018667, 0.0020062, 0.0018830, 0.0019929, -0.0000742, 0.0000708
2: 0.0117426, 0.0122764, 0.0117937, 0.0122142, -0.0002711, 0.0002838
3: -0.0025357, -0.0019836, -0.0024829, -0.0020480, -0.0002804, 0.0002935
4: -0.0018895, -0.0012919, -0.0018199, -0.0013491, -0.0003178, 0.0003035
5: 0.0053359, 0.0059015, 0.0053901, 0.0058356, -0.0002872, 0.0003007
6: -0.0011290, 0.0011151, -0.0009141, 0.0008537, -0.0011396, 0.0011932
7: -0.0040754, -0.0010191, -0.0037193, -0.0013118, -0.0016250, 0.0015520
8: 0.9863431, 0.9884960, 0.9865939, 0.9882898, -0.0011447, 0.0010933
9: -0.0054447, -0.0034904, -0.0052575, -0.0037181, -0.0009924, 0.0010391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005110
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005110
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0038758, 0.0046572, -0.0005622, 0.0005280
1: 0.0018667, 0.0020062, 0.0018822, 0.0019951, -0.0000812, 0.0000763
2: 0.0117426, 0.0122764, 0.0117850, 0.0122170, -0.0002919, 0.0003108
3: -0.0025357, -0.0019836, -0.0024918, -0.0020450, -0.0003019, 0.0003214
4: -0.0018895, -0.0012919, -0.0018231, -0.0013394, -0.0003480, 0.0003268
5: 0.0053359, 0.0059015, 0.0053809, 0.0058386, -0.0003093, 0.0003293
6: -0.0011290, 0.0011151, -0.0009506, 0.0008656, -0.0012272, 0.0013066
7: -0.0040754, -0.0010191, -0.0037356, -0.0012621, -0.0017795, 0.0016713
8: 0.9863431, 0.9884960, 0.9865823, 0.9883248, -0.0012535, 0.0011773
9: -0.0054447, -0.0034904, -0.0052894, -0.0037077, -0.0010687, 0.0011379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005246
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005246
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0038809, 0.0046415, -0.0006126, 0.0005523
1: 0.0018544, 0.0020129, 0.0018830, 0.0019929, -0.0000885, 0.0000798
2: 0.0117170, 0.0123237, 0.0117937, 0.0122142, -0.0003054, 0.0003387
3: -0.0025622, -0.0019347, -0.0024829, -0.0020480, -0.0003158, 0.0003503
4: -0.0019425, -0.0012632, -0.0018199, -0.0013491, -0.0003792, 0.0003419
5: 0.0053088, 0.0059516, 0.0053901, 0.0058356, -0.0003235, 0.0003589
6: -0.0012366, 0.0013139, -0.0009141, 0.0008537, -0.0012837, 0.0014238
7: -0.0043461, -0.0008726, -0.0037193, -0.0013118, -0.0019392, 0.0017484
8: 0.9861523, 0.9885992, 0.9865939, 0.9882898, -0.0013660, 0.0012316
9: -0.0055384, -0.0033173, -0.0052575, -0.0037181, -0.0011179, 0.0012399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0004940
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0004940
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0038758, 0.0046572, -0.0006614, 0.0005900
1: 0.0018544, 0.0020129, 0.0018822, 0.0019951, -0.0000956, 0.0000852
2: 0.0117170, 0.0123237, 0.0117850, 0.0122170, -0.0003262, 0.0003657
3: -0.0025622, -0.0019347, -0.0024918, -0.0020450, -0.0003374, 0.0003782
4: -0.0019425, -0.0012632, -0.0018231, -0.0013394, -0.0004094, 0.0003652
5: 0.0053088, 0.0059516, 0.0053809, 0.0058386, -0.0003456, 0.0003874
6: -0.0012366, 0.0013139, -0.0009506, 0.0008656, -0.0013714, 0.0015373
7: -0.0043461, -0.0008726, -0.0037356, -0.0012621, -0.0020936, 0.0018677
8: 0.9861523, 0.9885992, 0.9865823, 0.9883248, -0.0014748, 0.0013156
9: -0.0055384, -0.0033173, -0.0052894, -0.0037077, -0.0011942, 0.0013387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0005013
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0005013
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0037684, 0.0047340, -0.0004225, 0.0004225
1: 0.0018667, 0.0020062, 0.0018667, 0.0020062, -0.0000610, 0.0000610
2: 0.0117426, 0.0122764, 0.0117426, 0.0122764, -0.0002336, 0.0002336
3: -0.0025357, -0.0019836, -0.0025357, -0.0019836, -0.0002416, 0.0002416
4: -0.0018895, -0.0012919, -0.0018895, -0.0012919, -0.0002615, 0.0002615
5: 0.0053359, 0.0059015, 0.0053359, 0.0059015, -0.0002475, 0.0002475
6: -0.0011290, 0.0011151, -0.0011290, 0.0011151, -0.0009820, 0.0009820
7: -0.0040754, -0.0010191, -0.0040754, -0.0010191, -0.0013374, 0.0013374
8: 0.9863431, 0.9884960, 0.9863431, 0.9884960, -0.0009421, 0.0009421
9: -0.0054447, -0.0034904, -0.0054447, -0.0034904, -0.0008551, 0.0008551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005110
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005110
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0037654, 0.0047482, -0.0004699, 0.0004562
1: 0.0018667, 0.0020062, 0.0018663, 0.0020083, -0.0000679, 0.0000659
2: 0.0117426, 0.0122764, 0.0117347, 0.0122781, -0.0002522, 0.0002598
3: -0.0025357, -0.0019836, -0.0025439, -0.0019819, -0.0002609, 0.0002687
4: -0.0018895, -0.0012919, -0.0018914, -0.0012831, -0.0002909, 0.0002824
5: 0.0053359, 0.0059015, 0.0053276, 0.0059033, -0.0002673, 0.0002753
6: -0.0011290, 0.0011151, -0.0011620, 0.0011222, -0.0010604, 0.0010922
7: -0.0040754, -0.0010191, -0.0040850, -0.0009741, -0.0014875, 0.0014442
8: 0.9863431, 0.9884960, 0.9863362, 0.9885277, -0.0010478, 0.0010173
9: -0.0054447, -0.0034904, -0.0054735, -0.0034843, -0.0009235, 0.0009511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005247
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005247
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0037684, 0.0047340, -0.0005367, 0.0005039
1: 0.0018544, 0.0020129, 0.0018667, 0.0020062, -0.0000775, 0.0000728
2: 0.0117170, 0.0123237, 0.0117426, 0.0122764, -0.0002786, 0.0002967
3: -0.0025622, -0.0019347, -0.0025357, -0.0019836, -0.0002882, 0.0003069
4: -0.0019425, -0.0012632, -0.0018895, -0.0012919, -0.0003322, 0.0003119
5: 0.0053088, 0.0059516, 0.0053359, 0.0059015, -0.0002952, 0.0003144
6: -0.0012366, 0.0013139, -0.0011290, 0.0011151, -0.0011713, 0.0012474
7: -0.0043461, -0.0008726, -0.0040754, -0.0010191, -0.0016988, 0.0015952
8: 0.9861523, 0.9885992, 0.9863431, 0.9884960, -0.0011967, 0.0011237
9: -0.0055384, -0.0033173, -0.0054447, -0.0034904, -0.0010200, 0.0010863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0004940
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0004940
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0037654, 0.0047482, -0.0005841, 0.0005377
1: 0.0018544, 0.0020129, 0.0018663, 0.0020083, -0.0000844, 0.0000777
2: 0.0117170, 0.0123237, 0.0117347, 0.0122781, -0.0002973, 0.0003229
3: -0.0025622, -0.0019347, -0.0025439, -0.0019819, -0.0003075, 0.0003340
4: -0.0019425, -0.0012632, -0.0018914, -0.0012831, -0.0003616, 0.0003328
5: 0.0053088, 0.0059516, 0.0053276, 0.0059033, -0.0003150, 0.0003422
6: -0.0012366, 0.0013139, -0.0011620, 0.0011222, -0.0012497, 0.0013576
7: -0.0043461, -0.0008726, -0.0040850, -0.0009741, -0.0018489, 0.0017020
8: 0.9861523, 0.9885992, 0.9863362, 0.9885277, -0.0013024, 0.0011990
9: -0.0055384, -0.0033173, -0.0054735, -0.0034843, -0.0010883, 0.0011822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0005013
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0005013
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0038809, 0.0046415, -0.0005163, 0.0005064
1: 0.0018663, 0.0020083, 0.0018830, 0.0019929, -0.0000746, 0.0000732
2: 0.0117347, 0.0122781, 0.0117937, 0.0122142, -0.0002800, 0.0002855
3: -0.0025439, -0.0019819, -0.0024829, -0.0020480, -0.0002896, 0.0002952
4: -0.0018914, -0.0012831, -0.0018199, -0.0013491, -0.0003196, 0.0003135
5: 0.0053276, 0.0059033, 0.0053901, 0.0058356, -0.0002967, 0.0003025
6: -0.0011620, 0.0011222, -0.0009141, 0.0008537, -0.0011770, 0.0012001
7: -0.0040850, -0.0009741, -0.0037193, -0.0013118, -0.0016345, 0.0016030
8: 0.9863362, 0.9885277, 0.9865939, 0.9882898, -0.0011513, 0.0011292
9: -0.0054735, -0.0034843, -0.0052575, -0.0037181, -0.0010250, 0.0010451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0038758, 0.0046572, -0.0005226, 0.0004995
1: 0.0018663, 0.0020083, 0.0018822, 0.0019951, -0.0000755, 0.0000722
2: 0.0117347, 0.0122781, 0.0117850, 0.0122170, -0.0002762, 0.0002889
3: -0.0025439, -0.0019819, -0.0024918, -0.0020450, -0.0002856, 0.0002988
4: -0.0018914, -0.0012831, -0.0018231, -0.0013394, -0.0003235, 0.0003092
5: 0.0053276, 0.0059033, 0.0053809, 0.0058386, -0.0002926, 0.0003061
6: -0.0011620, 0.0011222, -0.0009506, 0.0008656, -0.0011611, 0.0012146
7: -0.0040850, -0.0009741, -0.0037356, -0.0012621, -0.0016541, 0.0015813
8: 0.9863362, 0.9885277, 0.9865823, 0.9883248, -0.0011652, 0.0011139
9: -0.0054735, -0.0034843, -0.0052894, -0.0037077, -0.0010111, 0.0010577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0038809, 0.0046415, -0.0006103, 0.0005638
1: 0.0018536, 0.0020156, 0.0018830, 0.0019929, -0.0000882, 0.0000815
2: 0.0117068, 0.0123267, 0.0117937, 0.0122142, -0.0003117, 0.0003374
3: -0.0025727, -0.0019316, -0.0024829, -0.0020480, -0.0003224, 0.0003490
4: -0.0019459, -0.0012518, -0.0018199, -0.0013491, -0.0003778, 0.0003490
5: 0.0052980, 0.0059548, 0.0053901, 0.0058356, -0.0003303, 0.0003575
6: -0.0012793, 0.0013267, -0.0009141, 0.0008537, -0.0013105, 0.0014185
7: -0.0043636, -0.0008144, -0.0037193, -0.0013118, -0.0019319, 0.0017848
8: 0.9861401, 0.9886402, 0.9865939, 0.9882898, -0.0013608, 0.0012573
9: -0.0055756, -0.0033061, -0.0052575, -0.0037181, -0.0011413, 0.0012353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0038758, 0.0046572, -0.0006219, 0.0005617
1: 0.0018536, 0.0020156, 0.0018822, 0.0019951, -0.0000898, 0.0000811
2: 0.0117068, 0.0123267, 0.0117850, 0.0122170, -0.0003105, 0.0003438
3: -0.0025727, -0.0019316, -0.0024918, -0.0020450, -0.0003212, 0.0003556
4: -0.0019459, -0.0012518, -0.0018231, -0.0013394, -0.0003850, 0.0003477
5: 0.0052980, 0.0059548, 0.0053809, 0.0058386, -0.0003290, 0.0003643
6: -0.0012793, 0.0013267, -0.0009506, 0.0008656, -0.0013055, 0.0014455
7: -0.0043636, -0.0008144, -0.0037356, -0.0012621, -0.0019686, 0.0017779
8: 0.9861401, 0.9886402, 0.9865823, 0.9883248, -0.0013867, 0.0012524
9: -0.0055756, -0.0033061, -0.0052894, -0.0037077, -0.0011369, 0.0012588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0037684, 0.0047340, -0.0004562, 0.0004699
1: 0.0018663, 0.0020083, 0.0018667, 0.0020062, -0.0000659, 0.0000679
2: 0.0117347, 0.0122781, 0.0117426, 0.0122764, -0.0002598, 0.0002522
3: -0.0025439, -0.0019819, -0.0025357, -0.0019836, -0.0002687, 0.0002609
4: -0.0018914, -0.0012831, -0.0018895, -0.0012919, -0.0002824, 0.0002909
5: 0.0053276, 0.0059033, 0.0053359, 0.0059015, -0.0002753, 0.0002673
6: -0.0011620, 0.0011222, -0.0011290, 0.0011151, -0.0010922, 0.0010604
7: -0.0040850, -0.0009741, -0.0040754, -0.0010191, -0.0014442, 0.0014875
8: 0.9863362, 0.9885277, 0.9863431, 0.9884960, -0.0010173, 0.0010478
9: -0.0054735, -0.0034843, -0.0054447, -0.0034904, -0.0009511, 0.0009235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0037654, 0.0047482, -0.0004287, 0.0004287
1: 0.0018663, 0.0020083, 0.0018663, 0.0020083, -0.0000619, 0.0000619
2: 0.0117347, 0.0122781, 0.0117347, 0.0122781, -0.0002370, 0.0002370
3: -0.0025439, -0.0019819, -0.0025439, -0.0019819, -0.0002452, 0.0002452
4: -0.0018914, -0.0012831, -0.0018914, -0.0012831, -0.0002654, 0.0002654
5: 0.0053276, 0.0059033, 0.0053276, 0.0059033, -0.0002512, 0.0002512
6: -0.0011620, 0.0011222, -0.0011620, 0.0011222, -0.0009965, 0.0009965
7: -0.0040850, -0.0009741, -0.0040850, -0.0009741, -0.0013571, 0.0013571
8: 0.9863362, 0.9885277, 0.9863362, 0.9885277, -0.0009560, 0.0009560
9: -0.0054735, -0.0034843, -0.0054735, -0.0034843, -0.0008678, 0.0008678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0037684, 0.0047340, -0.0005626, 0.0005411
1: 0.0018536, 0.0020156, 0.0018667, 0.0020062, -0.0000813, 0.0000782
2: 0.0117068, 0.0123267, 0.0117426, 0.0122764, -0.0002992, 0.0003111
3: -0.0025727, -0.0019316, -0.0025357, -0.0019836, -0.0003094, 0.0003217
4: -0.0019459, -0.0012518, -0.0018895, -0.0012919, -0.0003483, 0.0003350
5: 0.0052980, 0.0059548, 0.0053359, 0.0059015, -0.0003170, 0.0003296
6: -0.0012793, 0.0013267, -0.0011290, 0.0011151, -0.0012578, 0.0013077
7: -0.0043636, -0.0008144, -0.0040754, -0.0010191, -0.0017809, 0.0017130
8: 0.9861401, 0.9886402, 0.9863431, 0.9884960, -0.0012545, 0.0012067
9: -0.0055756, -0.0033061, -0.0054447, -0.0034904, -0.0010953, 0.0011388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006819, upper bound: 0.0004940
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0037654, 0.0047482, -0.0005430, 0.0005100
1: 0.0018536, 0.0020156, 0.0018663, 0.0020083, -0.0000784, 0.0000737
2: 0.0117068, 0.0123267, 0.0117347, 0.0122781, -0.0002820, 0.0003002
3: -0.0025727, -0.0019316, -0.0025439, -0.0019819, -0.0002916, 0.0003105
4: -0.0019459, -0.0012518, -0.0018914, -0.0012831, -0.0003361, 0.0003157
5: 0.0052980, 0.0059548, 0.0053276, 0.0059033, -0.0002987, 0.0003181
6: -0.0012793, 0.0013267, -0.0011620, 0.0011222, -0.0011853, 0.0012621
7: -0.0043636, -0.0008144, -0.0040850, -0.0009741, -0.0017188, 0.0016143
8: 0.9861401, 0.9886402, 0.9863362, 0.9885277, -0.0012108, 0.0011372
9: -0.0055756, -0.0033061, -0.0054735, -0.0034843, -0.0010323, 0.0010991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006819, upper bound: 0.0004940
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0037917, 0.0046918, -0.0005918, 0.0006035
1: 0.0018667, 0.0020062, 0.0018701, 0.0020001, -0.0000855, 0.0000872
2: 0.0117426, 0.0122764, 0.0117659, 0.0122635, -0.0003337, 0.0003272
3: -0.0025357, -0.0019836, -0.0025116, -0.0019969, -0.0003451, 0.0003384
4: -0.0018895, -0.0012919, -0.0018752, -0.0013180, -0.0003663, 0.0003736
5: 0.0053359, 0.0059015, 0.0053606, 0.0058879, -0.0003535, 0.0003467
6: -0.0011290, 0.0011151, -0.0010310, 0.0010610, -0.0014028, 0.0013755
7: -0.0040754, -0.0010191, -0.0040017, -0.0011526, -0.0018733, 0.0019105
8: 0.9863431, 0.9884960, 0.9863949, 0.9884019, -0.0013196, 0.0013458
9: -0.0054447, -0.0034904, -0.0053594, -0.0035375, -0.0012216, 0.0011979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006064, upper bound: 0.0005583
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006056, upper bound: 0.0005583
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0037879, 0.0047060, -0.0006299, 0.0006301
1: 0.0018667, 0.0020062, 0.0018695, 0.0020022, -0.0000910, 0.0000910
2: 0.0117426, 0.0122764, 0.0117580, 0.0122656, -0.0003484, 0.0003482
3: -0.0025357, -0.0019836, -0.0025197, -0.0019948, -0.0003603, 0.0003602
4: -0.0018895, -0.0012919, -0.0018775, -0.0013092, -0.0003899, 0.0003900
5: 0.0053359, 0.0059015, 0.0053523, 0.0058901, -0.0003691, 0.0003690
6: -0.0011290, 0.0011151, -0.0010640, 0.0010699, -0.0014645, 0.0014640
7: -0.0040754, -0.0010191, -0.0040138, -0.0011076, -0.0019939, 0.0019945
8: 0.9863431, 0.9884960, 0.9863865, 0.9884337, -0.0014045, 0.0014050
9: -0.0054447, -0.0034904, -0.0053881, -0.0035298, -0.0012754, 0.0012749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006064, upper bound: 0.0005734
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006056, upper bound: 0.0005734
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0037917, 0.0046918, -0.0005397, 0.0005162
1: 0.0018544, 0.0020129, 0.0018701, 0.0020001, -0.0000780, 0.0000746
2: 0.0117170, 0.0123237, 0.0117659, 0.0122635, -0.0002854, 0.0002984
3: -0.0025622, -0.0019347, -0.0025116, -0.0019969, -0.0002951, 0.0003086
4: -0.0019425, -0.0012632, -0.0018752, -0.0013180, -0.0003341, 0.0003195
5: 0.0053088, 0.0059516, 0.0053606, 0.0058879, -0.0003024, 0.0003161
6: -0.0012366, 0.0013139, -0.0010310, 0.0010610, -0.0011997, 0.0012544
7: -0.0043461, -0.0008726, -0.0040017, -0.0011526, -0.0017083, 0.0016339
8: 0.9861523, 0.9885992, 0.9863949, 0.9884019, -0.0012034, 0.0011510
9: -0.0055384, -0.0033173, -0.0053594, -0.0035375, -0.0010448, 0.0010923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0004940
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0004940
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0037879, 0.0047060, -0.0005875, 0.0005533
1: 0.0018544, 0.0020129, 0.0018695, 0.0020022, -0.0000849, 0.0000799
2: 0.0117170, 0.0123237, 0.0117580, 0.0122656, -0.0003059, 0.0003248
3: -0.0025622, -0.0019347, -0.0025197, -0.0019948, -0.0003164, 0.0003360
4: -0.0019425, -0.0012632, -0.0018775, -0.0013092, -0.0003637, 0.0003425
5: 0.0053088, 0.0059516, 0.0053523, 0.0058901, -0.0003241, 0.0003442
6: -0.0012366, 0.0013139, -0.0010640, 0.0010699, -0.0012861, 0.0013656
7: -0.0043461, -0.0008726, -0.0040138, -0.0011076, -0.0018598, 0.0017516
8: 0.9861523, 0.9885992, 0.9863865, 0.9884337, -0.0013101, 0.0012338
9: -0.0055384, -0.0033173, -0.0053881, -0.0035298, -0.0011200, 0.0011892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0005013
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0005013
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0036829, 0.0047802, -0.0005039, 0.0005367
1: 0.0018667, 0.0020062, 0.0018544, 0.0020129, -0.0000728, 0.0000775
2: 0.0117426, 0.0122764, 0.0117170, 0.0123237, -0.0002967, 0.0002786
3: -0.0025357, -0.0019836, -0.0025622, -0.0019347, -0.0003069, 0.0002882
4: -0.0018895, -0.0012919, -0.0019425, -0.0012632, -0.0003119, 0.0003322
5: 0.0053359, 0.0059015, 0.0053088, 0.0059516, -0.0003144, 0.0002952
6: -0.0011290, 0.0011151, -0.0012366, 0.0013139, -0.0012474, 0.0011713
7: -0.0040754, -0.0010191, -0.0043461, -0.0008726, -0.0015952, 0.0016988
8: 0.9863431, 0.9884960, 0.9861523, 0.9885992, -0.0011237, 0.0011967
9: -0.0054447, -0.0034904, -0.0055384, -0.0033173, -0.0010863, 0.0010200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006064, upper bound: 0.0005583
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006056, upper bound: 0.0005584
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037684, 0.0047340, 0.0036774, 0.0047986, -0.0005411, 0.0005626
1: 0.0018667, 0.0020062, 0.0018536, 0.0020156, -0.0000782, 0.0000813
2: 0.0117426, 0.0122764, 0.0117068, 0.0123267, -0.0003111, 0.0002992
3: -0.0025357, -0.0019836, -0.0025727, -0.0019316, -0.0003217, 0.0003094
4: -0.0018895, -0.0012919, -0.0019459, -0.0012518, -0.0003350, 0.0003483
5: 0.0053359, 0.0059015, 0.0052980, 0.0059548, -0.0003296, 0.0003170
6: -0.0011290, 0.0011151, -0.0012793, 0.0013267, -0.0013077, 0.0012578
7: -0.0040754, -0.0010191, -0.0043636, -0.0008144, -0.0017130, 0.0017809
8: 0.9863431, 0.9884960, 0.9861401, 0.9886402, -0.0012067, 0.0012545
9: -0.0054447, -0.0034904, -0.0055756, -0.0033061, -0.0011388, 0.0010953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006064, upper bound: 0.0005736
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006056, upper bound: 0.0005736
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0036829, 0.0047802, -0.0004495, 0.0004495
1: 0.0018544, 0.0020129, 0.0018544, 0.0020129, -0.0000649, 0.0000649
2: 0.0117170, 0.0123237, 0.0117170, 0.0123237, -0.0002485, 0.0002485
3: -0.0025622, -0.0019347, -0.0025622, -0.0019347, -0.0002570, 0.0002570
4: -0.0019425, -0.0012632, -0.0019425, -0.0012632, -0.0002783, 0.0002783
5: 0.0053088, 0.0059516, 0.0053088, 0.0059516, -0.0002633, 0.0002633
6: -0.0012366, 0.0013139, -0.0012366, 0.0013139, -0.0010448, 0.0010448
7: -0.0043461, -0.0008726, -0.0043461, -0.0008726, -0.0014229, 0.0014229
8: 0.9861523, 0.9885992, 0.9861523, 0.9885992, -0.0010023, 0.0010023
9: -0.0055384, -0.0033173, -0.0055384, -0.0033173, -0.0009099, 0.0009099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0004940
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0004940
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036829, 0.0047802, 0.0036774, 0.0047986, -0.0004960, 0.0004829
1: 0.0018544, 0.0020129, 0.0018536, 0.0020156, -0.0000717, 0.0000698
2: 0.0117170, 0.0123237, 0.0117068, 0.0123267, -0.0002670, 0.0002742
3: -0.0025622, -0.0019347, -0.0025727, -0.0019316, -0.0002761, 0.0002836
4: -0.0019425, -0.0012632, -0.0019459, -0.0012518, -0.0003070, 0.0002989
5: 0.0053088, 0.0059516, 0.0052980, 0.0059548, -0.0002829, 0.0002905
6: -0.0012366, 0.0013139, -0.0012793, 0.0013267, -0.0011224, 0.0011527
7: -0.0043461, -0.0008726, -0.0043636, -0.0008144, -0.0015699, 0.0015286
8: 0.9861523, 0.9885992, 0.9861401, 0.9886402, -0.0011059, 0.0010768
9: -0.0055384, -0.0033173, -0.0055756, -0.0033061, -0.0009774, 0.0010039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0005013
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0005013
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0037917, 0.0046918, -0.0005948, 0.0006197
1: 0.0018663, 0.0020083, 0.0018701, 0.0020001, -0.0000859, 0.0000895
2: 0.0117347, 0.0122781, 0.0117659, 0.0122635, -0.0003426, 0.0003288
3: -0.0025439, -0.0019819, -0.0025116, -0.0019969, -0.0003543, 0.0003401
4: -0.0018914, -0.0012831, -0.0018752, -0.0013180, -0.0003682, 0.0003836
5: 0.0053276, 0.0059033, 0.0053606, 0.0058879, -0.0003630, 0.0003484
6: -0.0011620, 0.0011222, -0.0010310, 0.0010610, -0.0014402, 0.0013824
7: -0.0040850, -0.0009741, -0.0040017, -0.0011526, -0.0018828, 0.0019615
8: 0.9863362, 0.9885277, 0.9863949, 0.9884019, -0.0013262, 0.0013817
9: -0.0054735, -0.0034843, -0.0053594, -0.0035375, -0.0012542, 0.0012039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006048, upper bound: 0.0005583
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006023, upper bound: 0.0005583
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0037879, 0.0047060, -0.0006011, 0.0006120
1: 0.0018663, 0.0020083, 0.0018695, 0.0020022, -0.0000868, 0.0000884
2: 0.0117347, 0.0122781, 0.0117580, 0.0122656, -0.0003384, 0.0003323
3: -0.0025439, -0.0019819, -0.0025197, -0.0019948, -0.0003500, 0.0003437
4: -0.0018914, -0.0012831, -0.0018775, -0.0013092, -0.0003721, 0.0003788
5: 0.0053276, 0.0059033, 0.0053523, 0.0058901, -0.0003585, 0.0003521
6: -0.0011620, 0.0011222, -0.0010640, 0.0010699, -0.0014225, 0.0013971
7: -0.0040850, -0.0009741, -0.0040138, -0.0011076, -0.0019027, 0.0019373
8: 0.9863362, 0.9885277, 0.9863865, 0.9884337, -0.0013403, 0.0013647
9: -0.0054735, -0.0034843, -0.0053881, -0.0035298, -0.0012388, 0.0012167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006048, upper bound: 0.0005583
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006023, upper bound: 0.0005583
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0037917, 0.0046918, -0.0005427, 0.0005317
1: 0.0018536, 0.0020156, 0.0018701, 0.0020001, -0.0000784, 0.0000768
2: 0.0117068, 0.0123267, 0.0117659, 0.0122635, -0.0002940, 0.0003000
3: -0.0025727, -0.0019316, -0.0025116, -0.0019969, -0.0003040, 0.0003103
4: -0.0019459, -0.0012518, -0.0018752, -0.0013180, -0.0003359, 0.0003291
5: 0.0052980, 0.0059548, 0.0053606, 0.0058879, -0.0003115, 0.0003179
6: -0.0012793, 0.0013267, -0.0010310, 0.0010610, -0.0012359, 0.0012613
7: -0.0043636, -0.0008144, -0.0040017, -0.0011526, -0.0017178, 0.0016831
8: 0.9861401, 0.9886402, 0.9863949, 0.9884019, -0.0012100, 0.0011856
9: -0.0055756, -0.0033061, -0.0053594, -0.0035375, -0.0010762, 0.0010984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0037879, 0.0047060, -0.0005491, 0.0005256
1: 0.0018536, 0.0020156, 0.0018695, 0.0020022, -0.0000793, 0.0000759
2: 0.0117068, 0.0123267, 0.0117580, 0.0122656, -0.0002906, 0.0003036
3: -0.0025727, -0.0019316, -0.0025197, -0.0019948, -0.0003006, 0.0003140
4: -0.0019459, -0.0012518, -0.0018775, -0.0013092, -0.0003399, 0.0003254
5: 0.0052980, 0.0059548, 0.0053523, 0.0058901, -0.0003079, 0.0003217
6: -0.0012793, 0.0013267, -0.0010640, 0.0010699, -0.0012217, 0.0012764
7: -0.0043636, -0.0008144, -0.0040138, -0.0011076, -0.0017383, 0.0016639
8: 0.9861401, 0.9886402, 0.9863865, 0.9884337, -0.0012245, 0.0011721
9: -0.0055756, -0.0033061, -0.0053881, -0.0035298, -0.0010639, 0.0011115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0036829, 0.0047802, -0.0005377, 0.0005841
1: 0.0018663, 0.0020083, 0.0018544, 0.0020129, -0.0000777, 0.0000844
2: 0.0117347, 0.0122781, 0.0117170, 0.0123237, -0.0003229, 0.0002973
3: -0.0025439, -0.0019819, -0.0025622, -0.0019347, -0.0003340, 0.0003075
4: -0.0018914, -0.0012831, -0.0019425, -0.0012632, -0.0003328, 0.0003616
5: 0.0053276, 0.0059033, 0.0053088, 0.0059516, -0.0003422, 0.0003150
6: -0.0011620, 0.0011222, -0.0012366, 0.0013139, -0.0013576, 0.0012497
7: -0.0040850, -0.0009741, -0.0043461, -0.0008726, -0.0017020, 0.0018489
8: 0.9863362, 0.9885277, 0.9861523, 0.9885992, -0.0011990, 0.0013024
9: -0.0054735, -0.0034843, -0.0055384, -0.0033173, -0.0011822, 0.0010883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006048, upper bound: 0.0005583
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006023, upper bound: 0.0005584
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037654, 0.0047482, 0.0036774, 0.0047986, -0.0005100, 0.0005430
1: 0.0018663, 0.0020083, 0.0018536, 0.0020156, -0.0000737, 0.0000784
2: 0.0117347, 0.0122781, 0.0117068, 0.0123267, -0.0003002, 0.0002820
3: -0.0025439, -0.0019819, -0.0025727, -0.0019316, -0.0003105, 0.0002916
4: -0.0018914, -0.0012831, -0.0019459, -0.0012518, -0.0003157, 0.0003361
5: 0.0053276, 0.0059033, 0.0052980, 0.0059548, -0.0003181, 0.0002987
6: -0.0011620, 0.0011222, -0.0012793, 0.0013267, -0.0012621, 0.0011853
7: -0.0040850, -0.0009741, -0.0043636, -0.0008144, -0.0016143, 0.0017188
8: 0.9863362, 0.9885277, 0.9861401, 0.9886402, -0.0011372, 0.0012108
9: -0.0054735, -0.0034843, -0.0055756, -0.0033061, -0.0010991, 0.0010323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006048, upper bound: 0.0005583
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006023, upper bound: 0.0005584
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0036829, 0.0047802, -0.0004829, 0.0004960
1: 0.0018536, 0.0020156, 0.0018544, 0.0020129, -0.0000698, 0.0000717
2: 0.0117068, 0.0123267, 0.0117170, 0.0123237, -0.0002742, 0.0002670
3: -0.0025727, -0.0019316, -0.0025622, -0.0019347, -0.0002836, 0.0002761
4: -0.0019459, -0.0012518, -0.0019425, -0.0012632, -0.0002989, 0.0003070
5: 0.0052980, 0.0059548, 0.0053088, 0.0059516, -0.0002905, 0.0002829
6: -0.0012793, 0.0013267, -0.0012366, 0.0013139, -0.0011527, 0.0011224
7: -0.0043636, -0.0008144, -0.0043461, -0.0008726, -0.0015286, 0.0015699
8: 0.9861401, 0.9886402, 0.9861523, 0.9885992, -0.0010768, 0.0011059
9: -0.0055756, -0.0033061, -0.0055384, -0.0033173, -0.0010039, 0.0009774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0036774, 0.0047986, 0.0036774, 0.0047986, -0.0004577, 0.0004577
1: 0.0018536, 0.0020156, 0.0018536, 0.0020156, -0.0000661, 0.0000661
2: 0.0117068, 0.0123267, 0.0117068, 0.0123267, -0.0002531, 0.0002531
3: -0.0025727, -0.0019316, -0.0025727, -0.0019316, -0.0002617, 0.0002617
4: -0.0019459, -0.0012518, -0.0019459, -0.0012518, -0.0002833, 0.0002833
5: 0.0052980, 0.0059548, 0.0052980, 0.0059548, -0.0002681, 0.0002681
6: -0.0012793, 0.0013267, -0.0012793, 0.0013267, -0.0010638, 0.0010638
7: -0.0043636, -0.0008144, -0.0043636, -0.0008144, -0.0014489, 0.0014489
8: 0.9861401, 0.9886402, 0.9861401, 0.9886402, -0.0010206, 0.0010206
9: -0.0055756, -0.0033061, -0.0055756, -0.0033061, -0.0009264, 0.0009264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
time: 0.51 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.69 seconds
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005082, upper bound: 0.0006176
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005111, upper bound: 0.0006176
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005082, upper bound: 0.0006238
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005111, upper bound: 0.0006238
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005542, upper bound: 0.0006010
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005586, upper bound: 0.0006012
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005542, upper bound: 0.0006019
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005586, upper bound: 0.0006024
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005245, upper bound: 0.0006176
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005246, upper bound: 0.0006176
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005245, upper bound: 0.0006176
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005246, upper bound: 0.0006176
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005719, upper bound: 0.0006010
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005734, upper bound: 0.0006012
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005719, upper bound: 0.0006010
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005734, upper bound: 0.0006012
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005142, upper bound: 0.0005836
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005191, upper bound: 0.0005836
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005142, upper bound: 0.0005934
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005191, upper bound: 0.0005934
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0004955, upper bound: 0.0006730
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0004981, upper bound: 0.0006730
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0004955, upper bound: 0.0006818
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0004981, upper bound: 0.0006820
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005542, upper bound: 0.0006010
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005586, upper bound: 0.0006012
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005542, upper bound: 0.0006019
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005586, upper bound: 0.0006024
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005207, upper bound: 0.0005836
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005212, upper bound: 0.0005836
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005207, upper bound: 0.0005836
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005212, upper bound: 0.0005836
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005036, upper bound: 0.0006730
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005012, upper bound: 0.0006730
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005036, upper bound: 0.0006730
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005012, upper bound: 0.0006730
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005719, upper bound: 0.0006010
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005734, upper bound: 0.0006012
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005719, upper bound: 0.0006010
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0005734, upper bound: 0.0006012
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005110
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005110
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005246
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005246
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0004940
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0004940
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0005013
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0005013
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005110
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005110
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005247
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006199, upper bound: 0.0005247
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0004940
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0004940
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0005013
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0005013
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006237, upper bound: 0.0005110
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006819, upper bound: 0.0004940
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006819, upper bound: 0.0004940
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006064, upper bound: 0.0005583
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006056, upper bound: 0.0005583
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006064, upper bound: 0.0005734
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006056, upper bound: 0.0005734
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0004940
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0004940
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0005013
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0005013
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006064, upper bound: 0.0005583
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006056, upper bound: 0.0005584
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006064, upper bound: 0.0005736
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006056, upper bound: 0.0005736
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0004940
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0004940
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006718, upper bound: 0.0005013
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006747, upper bound: 0.0005013
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006048, upper bound: 0.0005583
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006023, upper bound: 0.0005583
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006048, upper bound: 0.0005583
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006023, upper bound: 0.0005583
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006048, upper bound: 0.0005583
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006023, upper bound: 0.0005584
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006048, upper bound: 0.0005583
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006023, upper bound: 0.0005584
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006827, upper bound: 0.0004940
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 8, lower bound: -0.0006820, upper bound: 0.0004940

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038944, 0.0046407, 0.0037719, 0.0047338, -0.0004758, 0.0005099
1: 0.0018849, 0.0019928, 0.0018672, 0.0020062, -0.0000687, 0.0000737
2: 0.0117941, 0.0122068, 0.0117427, 0.0122745, -0.0002819, 0.0002631
3: -0.0024824, -0.0020557, -0.0025356, -0.0019856, -0.0002916, 0.0002721
4: -0.0018116, -0.0013496, -0.0018874, -0.0012920, -0.0002945, 0.0003157
5: 0.0053905, 0.0058277, 0.0053360, 0.0058995, -0.0002987, 0.0002787
6: -0.0009124, 0.0008224, -0.0011286, 0.0011070, -0.0011852, 0.0011059
7: -0.0036767, -0.0013142, -0.0040643, -0.0010196, -0.0015062, 0.0016142
8: 0.9866240, 0.9882881, 0.9863509, 0.9884956, -0.0010610, 0.0011371
9: -0.0052560, -0.0037454, -0.0054444, -0.0034975, -0.0010322, 0.0009631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004766, upper bound: 0.0005926
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004769, upper bound: 0.0005808
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0039055, 0.0046692, 0.0037810, 0.0047334, -0.0004767, 0.0005381
1: 0.0018865, 0.0019969, 0.0018685, 0.0020061, -0.0000689, 0.0000777
2: 0.0117784, 0.0122006, 0.0117429, 0.0122694, -0.0002975, 0.0002636
3: -0.0024987, -0.0020620, -0.0025354, -0.0019908, -0.0003077, 0.0002726
4: -0.0018047, -0.0013320, -0.0018818, -0.0012922, -0.0002951, 0.0003331
5: 0.0053738, 0.0058212, 0.0053362, 0.0058941, -0.0003152, 0.0002793
6: -0.0009786, 0.0007965, -0.0011279, 0.0010859, -0.0012507, 0.0011081
7: -0.0036415, -0.0012240, -0.0040356, -0.0010207, -0.0015091, 0.0017033
8: 0.9866487, 0.9883516, 0.9863712, 0.9884949, -0.0010631, 0.0011999
9: -0.0053137, -0.0037679, -0.0054437, -0.0035159, -0.0010892, 0.0009650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004784, upper bound: 0.0005926
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004786, upper bound: 0.0005808
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038944, 0.0046407, 0.0037689, 0.0047480, -0.0004919, 0.0005132
1: 0.0018849, 0.0019928, 0.0018668, 0.0020082, -0.0000711, 0.0000741
2: 0.0117941, 0.0122068, 0.0117348, 0.0122761, -0.0002837, 0.0002720
3: -0.0024824, -0.0020557, -0.0025437, -0.0019839, -0.0002935, 0.0002813
4: -0.0018116, -0.0013496, -0.0018893, -0.0012832, -0.0003045, 0.0003177
5: 0.0053905, 0.0058277, 0.0053277, 0.0059012, -0.0003006, 0.0002882
6: -0.0009124, 0.0008224, -0.0011616, 0.0011140, -0.0011929, 0.0011434
7: -0.0036767, -0.0013142, -0.0040739, -0.0009747, -0.0015572, 0.0016246
8: 0.9866240, 0.9882881, 0.9863442, 0.9885273, -0.0010969, 0.0011444
9: -0.0052560, -0.0037454, -0.0054731, -0.0034914, -0.0010388, 0.0009957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004758, upper bound: 0.0005964
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004760, upper bound: 0.0005813
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0039055, 0.0046692, 0.0037778, 0.0047476, -0.0004928, 0.0005419
1: 0.0018865, 0.0019969, 0.0018681, 0.0020082, -0.0000712, 0.0000783
2: 0.0117784, 0.0122006, 0.0117351, 0.0122712, -0.0002996, 0.0002725
3: -0.0024987, -0.0020620, -0.0025435, -0.0019890, -0.0003099, 0.0002818
4: -0.0018047, -0.0013320, -0.0018838, -0.0012835, -0.0003051, 0.0003355
5: 0.0053738, 0.0058212, 0.0053279, 0.0058960, -0.0003175, 0.0002887
6: -0.0009786, 0.0007965, -0.0011607, 0.0010933, -0.0012596, 0.0011454
7: -0.0036415, -0.0012240, -0.0040457, -0.0009760, -0.0015600, 0.0017155
8: 0.9866487, 0.9883516, 0.9863639, 0.9885263, -0.0010989, 0.0012084
9: -0.0053137, -0.0037679, -0.0054723, -0.0035094, -0.0010969, 0.0009975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004776, upper bound: 0.0005964
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004776, upper bound: 0.0005813
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038058, 0.0046911, 0.0037719, 0.0047338, -0.0005913, 0.0005882
1: 0.0018721, 0.0020000, 0.0018672, 0.0020062, -0.0000854, 0.0000850
2: 0.0117663, 0.0122557, 0.0117427, 0.0122745, -0.0003252, 0.0003269
3: -0.0025112, -0.0020050, -0.0025356, -0.0019856, -0.0003364, 0.0003381
4: -0.0018664, -0.0013184, -0.0018874, -0.0012920, -0.0003660, 0.0003641
5: 0.0053610, 0.0058796, 0.0053360, 0.0058995, -0.0003446, 0.0003464
6: -0.0010295, 0.0010282, -0.0011286, 0.0011070, -0.0013672, 0.0013744
7: -0.0039570, -0.0011546, -0.0040643, -0.0010196, -0.0018718, 0.0018620
8: 0.9864264, 0.9884005, 0.9863509, 0.9884956, -0.0013185, 0.0013117
9: -0.0053580, -0.0035661, -0.0054444, -0.0034975, -0.0011906, 0.0011969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0005789
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0005703
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038185, 0.0047170, 0.0037810, 0.0047334, -0.0005915, 0.0006146
1: 0.0018740, 0.0020038, 0.0018685, 0.0020061, -0.0000854, 0.0000888
2: 0.0117519, 0.0122487, 0.0117429, 0.0122694, -0.0003398, 0.0003270
3: -0.0025261, -0.0020123, -0.0025354, -0.0019908, -0.0003514, 0.0003382
4: -0.0018586, -0.0013023, -0.0018818, -0.0012922, -0.0003661, 0.0003804
5: 0.0053458, 0.0058722, 0.0053362, 0.0058941, -0.0003600, 0.0003465
6: -0.0010897, 0.0009987, -0.0011279, 0.0010859, -0.0014284, 0.0013747
7: -0.0039169, -0.0010726, -0.0040356, -0.0010207, -0.0018722, 0.0019454
8: 0.9864547, 0.9884583, 0.9863712, 0.9884949, -0.0013188, 0.0013703
9: -0.0054105, -0.0035918, -0.0054437, -0.0035159, -0.0012439, 0.0011972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005276, upper bound: 0.0005789
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005703
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038058, 0.0046911, 0.0037689, 0.0047480, -0.0006074, 0.0005915
1: 0.0018721, 0.0020000, 0.0018668, 0.0020082, -0.0000878, 0.0000855
2: 0.0117663, 0.0122557, 0.0117348, 0.0122761, -0.0003270, 0.0003358
3: -0.0025112, -0.0020050, -0.0025437, -0.0019839, -0.0003382, 0.0003473
4: -0.0018664, -0.0013184, -0.0018893, -0.0012832, -0.0003760, 0.0003662
5: 0.0053610, 0.0058796, 0.0053277, 0.0059012, -0.0003465, 0.0003558
6: -0.0010295, 0.0010282, -0.0011616, 0.0011140, -0.0013748, 0.0014118
7: -0.0039570, -0.0011546, -0.0040739, -0.0009747, -0.0019228, 0.0018724
8: 0.9864264, 0.9884005, 0.9863442, 0.9885273, -0.0013544, 0.0013190
9: -0.0053580, -0.0035661, -0.0054731, -0.0034914, -0.0011973, 0.0012295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005250, upper bound: 0.0005753
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005250, upper bound: 0.0005649
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0038185, 0.0047170, 0.0037778, 0.0047476, -0.0006075, 0.0006184
1: 0.0018740, 0.0020038, 0.0018681, 0.0020082, -0.0000878, 0.0000893
2: 0.0117519, 0.0122487, 0.0117351, 0.0122712, -0.0003419, 0.0003359
3: -0.0025261, -0.0020123, -0.0025435, -0.0019890, -0.0003536, 0.0003474
4: -0.0018586, -0.0013023, -0.0018838, -0.0012835, -0.0003761, 0.0003828
5: 0.0053458, 0.0058722, 0.0053279, 0.0058960, -0.0003623, 0.0003559
6: -0.0010897, 0.0009987, -0.0011607, 0.0010933, -0.0014373, 0.0014120
7: -0.0039169, -0.0010726, -0.0040457, -0.0009760, -0.0019231, 0.0019575
8: 0.9864547, 0.9884583, 0.9863639, 0.9885263, -0.0013547, 0.0013789
9: -0.0054105, -0.0035918, -0.0054723, -0.0035094, -0.0012517, 0.0012297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005272, upper bound: 0.0005755
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005275, upper bound: 0.0005654
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038891, 0.0046562, 0.0037719, 0.0047338, -0.0005131, 0.0005583
1: 0.0018842, 0.0019950, 0.0018672, 0.0020062, -0.0000741, 0.0000807
2: 0.0117856, 0.0122097, 0.0117427, 0.0122745, -0.0003087, 0.0002837
3: -0.0024913, -0.0020526, -0.0025356, -0.0019856, -0.0003192, 0.0002934
4: -0.0018149, -0.0013400, -0.0018874, -0.0012920, -0.0003176, 0.0003456
5: 0.0053815, 0.0058308, 0.0053360, 0.0058995, -0.0003270, 0.0003006
6: -0.0009482, 0.0008348, -0.0011286, 0.0011070, -0.0012976, 0.0011927
7: -0.0036936, -0.0012653, -0.0040643, -0.0010196, -0.0016243, 0.0017672
8: 0.9866120, 0.9883226, 0.9863509, 0.9884956, -0.0011442, 0.0012449
9: -0.0052873, -0.0037346, -0.0054444, -0.0034975, -0.0011300, 0.0010386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005924
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004883, upper bound: 0.0005789
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0039002, 0.0046848, 0.0037810, 0.0047334, -0.0005141, 0.0005821
1: 0.0018858, 0.0019991, 0.0018685, 0.0020061, -0.0000743, 0.0000841
2: 0.0117698, 0.0122036, 0.0117429, 0.0122694, -0.0003218, 0.0002842
3: -0.0025076, -0.0020590, -0.0025354, -0.0019908, -0.0003328, 0.0002940
4: -0.0018080, -0.0013223, -0.0018818, -0.0012922, -0.0003182, 0.0003603
5: 0.0053647, 0.0058243, 0.0053362, 0.0058941, -0.0003410, 0.0003011
6: -0.0010147, 0.0008089, -0.0011279, 0.0010859, -0.0013529, 0.0011949
7: -0.0036584, -0.0011748, -0.0040356, -0.0010207, -0.0016273, 0.0018426
8: 0.9866368, 0.9883863, 0.9863712, 0.9884949, -0.0011463, 0.0012979
9: -0.0053452, -0.0037571, -0.0054437, -0.0035159, -0.0011782, 0.0010405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005924
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005789
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038891, 0.0046562, 0.0037689, 0.0047480, -0.0004849, 0.0005190
1: 0.0018842, 0.0019950, 0.0018668, 0.0020082, -0.0000701, 0.0000750
2: 0.0117856, 0.0122097, 0.0117348, 0.0122761, -0.0002869, 0.0002681
3: -0.0024913, -0.0020526, -0.0025437, -0.0019839, -0.0002968, 0.0002773
4: -0.0018149, -0.0013400, -0.0018893, -0.0012832, -0.0003002, 0.0003213
5: 0.0053815, 0.0058308, 0.0053277, 0.0059012, -0.0003040, 0.0002841
6: -0.0009482, 0.0008348, -0.0011616, 0.0011140, -0.0012063, 0.0011271
7: -0.0036936, -0.0012653, -0.0040739, -0.0009747, -0.0015350, 0.0016429
8: 0.9866120, 0.9883226, 0.9863442, 0.9885273, -0.0010813, 0.0011573
9: -0.0052873, -0.0037346, -0.0054731, -0.0034914, -0.0010505, 0.0009815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005906
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004883, upper bound: 0.0005758
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0039002, 0.0046848, 0.0037778, 0.0047476, -0.0004858, 0.0005477
1: 0.0018858, 0.0019991, 0.0018681, 0.0020082, -0.0000702, 0.0000791
2: 0.0117698, 0.0122036, 0.0117351, 0.0122712, -0.0003028, 0.0002686
3: -0.0025076, -0.0020590, -0.0025435, -0.0019890, -0.0003132, 0.0002778
4: -0.0018080, -0.0013223, -0.0018838, -0.0012835, -0.0003007, 0.0003390
5: 0.0053647, 0.0058243, 0.0053279, 0.0058960, -0.0003208, 0.0002846
6: -0.0010147, 0.0008089, -0.0011607, 0.0010933, -0.0012729, 0.0011290
7: -0.0036584, -0.0011748, -0.0040457, -0.0009760, -0.0015376, 0.0017336
8: 0.9866368, 0.9883863, 0.9863639, 0.9885263, -0.0010831, 0.0012212
9: -0.0053452, -0.0037571, -0.0054723, -0.0035094, -0.0011085, 0.0009832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005906
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005758
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038019, 0.0047052, 0.0037719, 0.0047338, -0.0006175, 0.0006260
1: 0.0018716, 0.0020021, 0.0018672, 0.0020062, -0.0000892, 0.0000904
2: 0.0117585, 0.0122579, 0.0117427, 0.0122745, -0.0003461, 0.0003414
3: -0.0025193, -0.0020028, -0.0025356, -0.0019856, -0.0003580, 0.0003531
4: -0.0018688, -0.0013097, -0.0018874, -0.0012920, -0.0003822, 0.0003875
5: 0.0053527, 0.0058819, 0.0053360, 0.0058995, -0.0003667, 0.0003617
6: -0.0010622, 0.0010373, -0.0011286, 0.0011070, -0.0014550, 0.0014352
7: -0.0039694, -0.0011101, -0.0040643, -0.0010196, -0.0019547, 0.0019816
8: 0.9864177, 0.9884319, 0.9863509, 0.9884956, -0.0013769, 0.0013959
9: -0.0053865, -0.0035582, -0.0054444, -0.0034975, -0.0012671, 0.0012499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005787
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005399, upper bound: 0.0005691
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038152, 0.0047304, 0.0037810, 0.0047334, -0.0006177, 0.0006462
1: 0.0018735, 0.0020057, 0.0018685, 0.0020061, -0.0000892, 0.0000934
2: 0.0117446, 0.0122505, 0.0117429, 0.0122694, -0.0003573, 0.0003415
3: -0.0025337, -0.0020104, -0.0025354, -0.0019908, -0.0003695, 0.0003532
4: -0.0018606, -0.0012941, -0.0018818, -0.0012922, -0.0003824, 0.0004000
5: 0.0053380, 0.0058741, 0.0053362, 0.0058941, -0.0003785, 0.0003618
6: -0.0011207, 0.0010064, -0.0011279, 0.0010859, -0.0015019, 0.0014356
7: -0.0039273, -0.0010304, -0.0040356, -0.0010207, -0.0019552, 0.0020454
8: 0.9864475, 0.9884880, 0.9863712, 0.9884949, -0.0013773, 0.0014408
9: -0.0054375, -0.0035851, -0.0054437, -0.0035159, -0.0013079, 0.0012502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005787
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005424, upper bound: 0.0005691
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038019, 0.0047052, 0.0037689, 0.0047480, -0.0005998, 0.0005974
1: 0.0018716, 0.0020021, 0.0018668, 0.0020082, -0.0000866, 0.0000863
2: 0.0117585, 0.0122579, 0.0117348, 0.0122761, -0.0003303, 0.0003316
3: -0.0025193, -0.0020028, -0.0025437, -0.0019839, -0.0003416, 0.0003429
4: -0.0018688, -0.0013097, -0.0018893, -0.0012832, -0.0003713, 0.0003698
5: 0.0053527, 0.0058819, 0.0053277, 0.0059012, -0.0003499, 0.0003513
6: -0.0010622, 0.0010373, -0.0011616, 0.0011140, -0.0013885, 0.0013940
7: -0.0039694, -0.0011101, -0.0040739, -0.0009747, -0.0018985, 0.0018910
8: 0.9864177, 0.9884319, 0.9863442, 0.9885273, -0.0013374, 0.0013320
9: -0.0053865, -0.0035582, -0.0054731, -0.0034914, -0.0012091, 0.0012140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005747
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005399, upper bound: 0.0005641
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0038152, 0.0047304, 0.0037778, 0.0047476, -0.0005999, 0.0006241
1: 0.0018735, 0.0020057, 0.0018681, 0.0020082, -0.0000867, 0.0000902
2: 0.0117446, 0.0122505, 0.0117351, 0.0122712, -0.0003451, 0.0003317
3: -0.0025337, -0.0020104, -0.0025435, -0.0019890, -0.0003569, 0.0003430
4: -0.0018606, -0.0012941, -0.0018838, -0.0012835, -0.0003713, 0.0003864
5: 0.0053380, 0.0058741, 0.0053279, 0.0058960, -0.0003656, 0.0003514
6: -0.0011207, 0.0010064, -0.0011607, 0.0010933, -0.0014507, 0.0013943
7: -0.0039273, -0.0010304, -0.0040457, -0.0009760, -0.0018990, 0.0019757
8: 0.9864475, 0.9884880, 0.9863639, 0.9885263, -0.0013377, 0.0013917
9: -0.0054375, -0.0035851, -0.0054723, -0.0035094, -0.0012633, 0.0012142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005747
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005424, upper bound: 0.0005641
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038944, 0.0046407, 0.0037915, 0.0047058, -0.0004800, 0.0005143
1: 0.0018849, 0.0019928, 0.0018701, 0.0020022, -0.0000693, 0.0000743
2: 0.0117941, 0.0122068, 0.0117581, 0.0122636, -0.0002844, 0.0002654
3: -0.0024824, -0.0020557, -0.0025196, -0.0019968, -0.0002941, 0.0002745
4: -0.0018116, -0.0013496, -0.0018753, -0.0013093, -0.0002971, 0.0003184
5: 0.0053905, 0.0058277, 0.0053524, 0.0058880, -0.0003013, 0.0002812
6: -0.0009124, 0.0008224, -0.0010636, 0.0010615, -0.0011955, 0.0011156
7: -0.0036767, -0.0013142, -0.0040024, -0.0011082, -0.0015193, 0.0016281
8: 0.9866240, 0.9882881, 0.9863945, 0.9884333, -0.0010703, 0.0011469
9: -0.0052560, -0.0037454, -0.0053877, -0.0035371, -0.0010411, 0.0009715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004822, upper bound: 0.0005686
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004833, upper bound: 0.0005658
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0039055, 0.0046692, 0.0038000, 0.0047055, -0.0004809, 0.0005444
1: 0.0018865, 0.0019969, 0.0018713, 0.0020021, -0.0000695, 0.0000787
2: 0.0117784, 0.0122006, 0.0117583, 0.0122589, -0.0003010, 0.0002659
3: -0.0024987, -0.0020620, -0.0025195, -0.0020017, -0.0003113, 0.0002750
4: -0.0018047, -0.0013320, -0.0018700, -0.0013095, -0.0002977, 0.0003370
5: 0.0053738, 0.0058212, 0.0053526, 0.0058830, -0.0003189, 0.0002817
6: -0.0009786, 0.0007965, -0.0010629, 0.0010418, -0.0012654, 0.0011178
7: -0.0036415, -0.0012240, -0.0039755, -0.0011091, -0.0015224, 0.0017233
8: 0.9866487, 0.9883516, 0.9864134, 0.9884326, -0.0010724, 0.0012139
9: -0.0053137, -0.0037679, -0.0053872, -0.0035543, -0.0011019, 0.0009734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004871, upper bound: 0.0005686
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004877, upper bound: 0.0005658
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038944, 0.0046407, 0.0036864, 0.0047801, -0.0005379, 0.0006090
1: 0.0018849, 0.0019928, 0.0018549, 0.0020129, -0.0000777, 0.0000880
2: 0.0117941, 0.0122068, 0.0117171, 0.0123217, -0.0003367, 0.0002974
3: -0.0024824, -0.0020557, -0.0025621, -0.0019367, -0.0003482, 0.0003076
4: -0.0018116, -0.0013496, -0.0019403, -0.0012633, -0.0003330, 0.0003770
5: 0.0053905, 0.0058277, 0.0053089, 0.0059495, -0.0003567, 0.0003151
6: -0.0009124, 0.0008224, -0.0012363, 0.0013057, -0.0014154, 0.0012502
7: -0.0036767, -0.0013142, -0.0043350, -0.0008730, -0.0017027, 0.0019276
8: 0.9866240, 0.9882881, 0.9861602, 0.9885989, -0.0011994, 0.0013579
9: -0.0052560, -0.0037454, -0.0055381, -0.0033244, -0.0012326, 0.0010887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004631, upper bound: 0.0006453
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004636, upper bound: 0.0006447
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0039055, 0.0046692, 0.0036952, 0.0047799, -0.0005389, 0.0006381
1: 0.0018865, 0.0019969, 0.0018562, 0.0020129, -0.0000779, 0.0000922
2: 0.0117784, 0.0122006, 0.0117172, 0.0123169, -0.0003528, 0.0002979
3: -0.0024987, -0.0020620, -0.0025620, -0.0019418, -0.0003649, 0.0003081
4: -0.0018047, -0.0013320, -0.0019349, -0.0012635, -0.0003336, 0.0003950
5: 0.0053738, 0.0058212, 0.0053090, 0.0059444, -0.0003738, 0.0003157
6: -0.0009786, 0.0007965, -0.0012357, 0.0012853, -0.0014831, 0.0012525
7: -0.0036415, -0.0012240, -0.0043072, -0.0008738, -0.0017059, 0.0020199
8: 0.9866487, 0.9883516, 0.9861797, 0.9885983, -0.0012016, 0.0014229
9: -0.0053137, -0.0037679, -0.0055376, -0.0033422, -0.0012916, 0.0010908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004680, upper bound: 0.0006453
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004680, upper bound: 0.0006447
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038944, 0.0046407, 0.0036808, 0.0047985, -0.0005494, 0.0006065
1: 0.0018849, 0.0019928, 0.0018541, 0.0020155, -0.0000794, 0.0000876
2: 0.0117941, 0.0122068, 0.0117069, 0.0123249, -0.0003353, 0.0003037
3: -0.0024824, -0.0020557, -0.0025726, -0.0019335, -0.0003468, 0.0003141
4: -0.0018116, -0.0013496, -0.0019438, -0.0012520, -0.0003401, 0.0003754
5: 0.0053905, 0.0058277, 0.0052981, 0.0059529, -0.0003553, 0.0003218
6: -0.0009124, 0.0008224, -0.0012790, 0.0013189, -0.0014096, 0.0012769
7: -0.0036767, -0.0013142, -0.0043529, -0.0008149, -0.0017390, 0.0019197
8: 0.9866240, 0.9882881, 0.9861476, 0.9886399, -0.0012250, 0.0013523
9: -0.0052560, -0.0037454, -0.0055753, -0.0033130, -0.0012275, 0.0011119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004629, upper bound: 0.0006534
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004631, upper bound: 0.0006502
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0039055, 0.0046692, 0.0036904, 0.0047981, -0.0005502, 0.0006380
1: 0.0018865, 0.0019969, 0.0018555, 0.0020155, -0.0000795, 0.0000922
2: 0.0117784, 0.0122006, 0.0117071, 0.0123195, -0.0003527, 0.0003042
3: -0.0024987, -0.0020620, -0.0025724, -0.0019390, -0.0003648, 0.0003146
4: -0.0018047, -0.0013320, -0.0019378, -0.0012522, -0.0003406, 0.0003949
5: 0.0053738, 0.0058212, 0.0052983, 0.0059472, -0.0003738, 0.0003223
6: -0.0009786, 0.0007965, -0.0012781, 0.0012964, -0.0014829, 0.0012789
7: -0.0036415, -0.0012240, -0.0043223, -0.0008160, -0.0017417, 0.0020196
8: 0.9866487, 0.9883516, 0.9861692, 0.9886391, -0.0012269, 0.0014227
9: -0.0053137, -0.0037679, -0.0055746, -0.0033326, -0.0012914, 0.0011137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004676, upper bound: 0.0006535
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004676, upper bound: 0.0006502
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038058, 0.0046911, 0.0036864, 0.0047801, -0.0005016, 0.0005362
1: 0.0018721, 0.0020000, 0.0018549, 0.0020129, -0.0000725, 0.0000775
2: 0.0117663, 0.0122557, 0.0117171, 0.0123217, -0.0002965, 0.0002773
3: -0.0025112, -0.0020050, -0.0025621, -0.0019367, -0.0003066, 0.0002868
4: -0.0018664, -0.0013184, -0.0019403, -0.0012633, -0.0003105, 0.0003319
5: 0.0053610, 0.0058796, 0.0053089, 0.0059495, -0.0003141, 0.0002938
6: -0.0010295, 0.0010282, -0.0012363, 0.0013057, -0.0012463, 0.0011658
7: -0.0039570, -0.0011546, -0.0043350, -0.0008730, -0.0015878, 0.0016974
8: 0.9864264, 0.9884005, 0.9861602, 0.9885989, -0.0011185, 0.0011957
9: -0.0053580, -0.0035661, -0.0055381, -0.0033244, -0.0010854, 0.0010153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005255, upper bound: 0.0005798
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005255, upper bound: 0.0005727
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038185, 0.0047170, 0.0036952, 0.0047799, -0.0005027, 0.0005641
1: 0.0018740, 0.0020038, 0.0018562, 0.0020129, -0.0000726, 0.0000815
2: 0.0117519, 0.0122487, 0.0117172, 0.0123169, -0.0003119, 0.0002779
3: -0.0025261, -0.0020123, -0.0025620, -0.0019418, -0.0003226, 0.0002875
4: -0.0018586, -0.0013023, -0.0019349, -0.0012635, -0.0003112, 0.0003492
5: 0.0053458, 0.0058722, 0.0053090, 0.0059444, -0.0003305, 0.0002945
6: -0.0010897, 0.0009987, -0.0012357, 0.0012853, -0.0013112, 0.0011685
7: -0.0039169, -0.0010726, -0.0043072, -0.0008738, -0.0015913, 0.0017857
8: 0.9864547, 0.9884583, 0.9861797, 0.9885983, -0.0011210, 0.0012579
9: -0.0054105, -0.0035918, -0.0055376, -0.0033422, -0.0011418, 0.0010176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005798
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005280, upper bound: 0.0005727
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038058, 0.0046911, 0.0036808, 0.0047985, -0.0005171, 0.0005395
1: 0.0018721, 0.0020000, 0.0018541, 0.0020155, -0.0000747, 0.0000779
2: 0.0117663, 0.0122557, 0.0117069, 0.0123249, -0.0002983, 0.0002859
3: -0.0025112, -0.0020050, -0.0025726, -0.0019335, -0.0003085, 0.0002957
4: -0.0018664, -0.0013184, -0.0019438, -0.0012520, -0.0003201, 0.0003340
5: 0.0053610, 0.0058796, 0.0052981, 0.0059529, -0.0003161, 0.0003029
6: -0.0010295, 0.0010282, -0.0012790, 0.0013189, -0.0012541, 0.0012020
7: -0.0039570, -0.0011546, -0.0043529, -0.0008149, -0.0016370, 0.0017079
8: 0.9864264, 0.9884005, 0.9861476, 0.9886399, -0.0011531, 0.0012031
9: -0.0053580, -0.0035661, -0.0055753, -0.0033130, -0.0010921, 0.0010467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005252, upper bound: 0.0005762
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005252, upper bound: 0.0005676
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0038185, 0.0047170, 0.0036904, 0.0047981, -0.0005182, 0.0005680
1: 0.0018740, 0.0020038, 0.0018555, 0.0020155, -0.0000749, 0.0000821
2: 0.0117519, 0.0122487, 0.0117071, 0.0123195, -0.0003140, 0.0002865
3: -0.0025261, -0.0020123, -0.0025724, -0.0019390, -0.0003248, 0.0002963
4: -0.0018586, -0.0013023, -0.0019378, -0.0012522, -0.0003208, 0.0003516
5: 0.0053458, 0.0058722, 0.0052983, 0.0059472, -0.0003327, 0.0003036
6: -0.0010897, 0.0009987, -0.0012781, 0.0012964, -0.0013202, 0.0012045
7: -0.0039169, -0.0010726, -0.0043223, -0.0008160, -0.0016404, 0.0017980
8: 0.9864547, 0.9884583, 0.9861692, 0.9886391, -0.0011555, 0.0012665
9: -0.0054105, -0.0035918, -0.0055746, -0.0033326, -0.0011497, 0.0010489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005275, upper bound: 0.0005765
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005684
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038891, 0.0046562, 0.0036864, 0.0047801, -0.0005752, 0.0006573
1: 0.0018842, 0.0019950, 0.0018549, 0.0020129, -0.0000831, 0.0000950
2: 0.0117856, 0.0122097, 0.0117171, 0.0123217, -0.0003634, 0.0003180
3: -0.0024913, -0.0020526, -0.0025621, -0.0019367, -0.0003758, 0.0003289
4: -0.0018149, -0.0013400, -0.0019403, -0.0012633, -0.0003561, 0.0004069
5: 0.0053815, 0.0058308, 0.0053089, 0.0059495, -0.0003850, 0.0003370
6: -0.0009482, 0.0008348, -0.0012363, 0.0013057, -0.0015277, 0.0013370
7: -0.0036936, -0.0012653, -0.0043350, -0.0008730, -0.0018208, 0.0020806
8: 0.9866120, 0.9883226, 0.9861602, 0.9885989, -0.0012826, 0.0014656
9: -0.0052873, -0.0037346, -0.0055381, -0.0033244, -0.0013304, 0.0011643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004711, upper bound: 0.0006451
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004715, upper bound: 0.0006441
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0039002, 0.0046848, 0.0036952, 0.0047799, -0.0005762, 0.0006821
1: 0.0018858, 0.0019991, 0.0018562, 0.0020129, -0.0000832, 0.0000985
2: 0.0117698, 0.0122036, 0.0117172, 0.0123169, -0.0003771, 0.0003186
3: -0.0025076, -0.0020590, -0.0025620, -0.0019418, -0.0003900, 0.0003295
4: -0.0018080, -0.0013223, -0.0019349, -0.0012635, -0.0003567, 0.0004222
5: 0.0053647, 0.0058243, 0.0053090, 0.0059444, -0.0003996, 0.0003376
6: -0.0010147, 0.0008089, -0.0012357, 0.0012853, -0.0015854, 0.0013393
7: -0.0036584, -0.0011748, -0.0043072, -0.0008738, -0.0018240, 0.0021591
8: 0.9866368, 0.9883863, 0.9861797, 0.9885983, -0.0012849, 0.0015209
9: -0.0053452, -0.0037571, -0.0055376, -0.0033422, -0.0013806, 0.0011663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006451
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006441
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038891, 0.0046562, 0.0036808, 0.0047985, -0.0005471, 0.0006182
1: 0.0018842, 0.0019950, 0.0018541, 0.0020155, -0.0000790, 0.0000893
2: 0.0117856, 0.0122097, 0.0117069, 0.0123249, -0.0003418, 0.0003025
3: -0.0024913, -0.0020526, -0.0025726, -0.0019335, -0.0003535, 0.0003128
4: -0.0018149, -0.0013400, -0.0019438, -0.0012520, -0.0003387, 0.0003827
5: 0.0053815, 0.0058308, 0.0052981, 0.0059529, -0.0003621, 0.0003205
6: -0.0009482, 0.0008348, -0.0012790, 0.0013189, -0.0014369, 0.0012716
7: -0.0036936, -0.0012653, -0.0043529, -0.0008149, -0.0017318, 0.0019569
8: 0.9866120, 0.9883226, 0.9861476, 0.9886399, -0.0012199, 0.0013785
9: -0.0052873, -0.0037346, -0.0055753, -0.0033130, -0.0012513, 0.0011073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004711, upper bound: 0.0006441
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004715, upper bound: 0.0006421
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0039002, 0.0046848, 0.0036904, 0.0047981, -0.0005480, 0.0006477
1: 0.0018858, 0.0019991, 0.0018555, 0.0020155, -0.0000792, 0.0000936
2: 0.0117698, 0.0122036, 0.0117071, 0.0123195, -0.0003581, 0.0003030
3: -0.0025076, -0.0020590, -0.0025724, -0.0019390, -0.0003704, 0.0003134
4: -0.0018080, -0.0013223, -0.0019378, -0.0012522, -0.0003392, 0.0004009
5: 0.0053647, 0.0058243, 0.0052983, 0.0059472, -0.0003794, 0.0003210
6: -0.0010147, 0.0008089, -0.0012781, 0.0012964, -0.0015055, 0.0012737
7: -0.0036584, -0.0011748, -0.0043223, -0.0008160, -0.0017347, 0.0020503
8: 0.9866368, 0.9883863, 0.9861692, 0.9886391, -0.0012219, 0.0014443
9: -0.0053452, -0.0037571, -0.0055746, -0.0033326, -0.0013110, 0.0011092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006441
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006421
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0038019, 0.0047052, 0.0036864, 0.0047801, -0.0005386, 0.0005837
1: 0.0018716, 0.0020021, 0.0018549, 0.0020129, -0.0000778, 0.0000843
2: 0.0117585, 0.0122579, 0.0117171, 0.0123217, -0.0003227, 0.0002978
3: -0.0025193, -0.0020028, -0.0025621, -0.0019367, -0.0003338, 0.0003080
4: -0.0018688, -0.0013097, -0.0019403, -0.0012633, -0.0003334, 0.0003613
5: 0.0053527, 0.0058819, 0.0053089, 0.0059495, -0.0003419, 0.0003155
6: -0.0010622, 0.0010373, -0.0012363, 0.0013057, -0.0013567, 0.0012519
7: -0.0039694, -0.0011101, -0.0043350, -0.0008730, -0.0017050, 0.0018477
8: 0.9864177, 0.9884319, 0.9861602, 0.9885989, -0.0012010, 0.0013015
9: -0.0053865, -0.0035582, -0.0055381, -0.0033244, -0.0011814, 0.0010902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005797
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005402, upper bound: 0.0005721
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038152, 0.0047304, 0.0036952, 0.0047799, -0.0005396, 0.0006076
1: 0.0018735, 0.0020057, 0.0018562, 0.0020129, -0.0000780, 0.0000878
2: 0.0117446, 0.0122505, 0.0117172, 0.0123169, -0.0003359, 0.0002983
3: -0.0025337, -0.0020104, -0.0025620, -0.0019418, -0.0003474, 0.0003086
4: -0.0018606, -0.0012941, -0.0019349, -0.0012635, -0.0003340, 0.0003761
5: 0.0053380, 0.0058741, 0.0053090, 0.0059444, -0.0003559, 0.0003161
6: -0.0011207, 0.0010064, -0.0012357, 0.0012853, -0.0014123, 0.0012542
7: -0.0039273, -0.0010304, -0.0043072, -0.0008738, -0.0017081, 0.0019234
8: 0.9864475, 0.9884880, 0.9861797, 0.9885983, -0.0012032, 0.0013549
9: -0.0054375, -0.0035851, -0.0055376, -0.0033422, -0.0012299, 0.0010922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005797
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005427, upper bound: 0.0005721
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0038019, 0.0047052, 0.0036808, 0.0047985, -0.0005110, 0.0005456
1: 0.0018716, 0.0020021, 0.0018541, 0.0020155, -0.0000738, 0.0000788
2: 0.0117585, 0.0122579, 0.0117069, 0.0123249, -0.0003017, 0.0002825
3: -0.0025193, -0.0020028, -0.0025726, -0.0019335, -0.0003120, 0.0002922
4: -0.0018688, -0.0013097, -0.0019438, -0.0012520, -0.0003163, 0.0003377
5: 0.0053527, 0.0058819, 0.0052981, 0.0059529, -0.0003196, 0.0002993
6: -0.0010622, 0.0010373, -0.0012790, 0.0013189, -0.0012681, 0.0011877
7: -0.0039694, -0.0011101, -0.0043529, -0.0008149, -0.0016175, 0.0017271
8: 0.9864177, 0.9884319, 0.9861476, 0.9886399, -0.0011394, 0.0012166
9: -0.0053865, -0.0035582, -0.0055753, -0.0033130, -0.0011044, 0.0010343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005757
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005402, upper bound: 0.0005669
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0038152, 0.0047304, 0.0036904, 0.0047981, -0.0005113, 0.0005740
1: 0.0018735, 0.0020057, 0.0018555, 0.0020155, -0.0000739, 0.0000829
2: 0.0117446, 0.0122505, 0.0117071, 0.0123195, -0.0003174, 0.0002827
3: -0.0025337, -0.0020104, -0.0025724, -0.0019390, -0.0003282, 0.0002924
4: -0.0018606, -0.0012941, -0.0019378, -0.0012522, -0.0003165, 0.0003553
5: 0.0053380, 0.0058741, 0.0052983, 0.0059472, -0.0003363, 0.0002995
6: -0.0011207, 0.0010064, -0.0012781, 0.0012964, -0.0013342, 0.0011885
7: -0.0039273, -0.0010304, -0.0043223, -0.0008160, -0.0016186, 0.0018170
8: 0.9864475, 0.9884880, 0.9861692, 0.9886391, -0.0011402, 0.0012800
9: -0.0054375, -0.0035851, -0.0055746, -0.0033326, -0.0011619, 0.0010350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005757
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005427, upper bound: 0.0005669
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037813, 0.0047333, 0.0038843, 0.0046413, -0.0005009, 0.0004860
1: 0.0018686, 0.0020061, 0.0018835, 0.0019928, -0.0000724, 0.0000702
2: 0.0117430, 0.0122693, 0.0117938, 0.0122123, -0.0002687, 0.0002769
3: -0.0025353, -0.0019910, -0.0024827, -0.0020499, -0.0002779, 0.0002864
4: -0.0018816, -0.0012923, -0.0018178, -0.0013492, -0.0003101, 0.0003009
5: 0.0053363, 0.0058939, 0.0053902, 0.0058336, -0.0002847, 0.0002934
6: -0.0011274, 0.0010851, -0.0009137, 0.0008457, -0.0011296, 0.0011642
7: -0.0040346, -0.0010212, -0.0037085, -0.0013124, -0.0015855, 0.0015385
8: 0.9863718, 0.9884945, 0.9866015, 0.9882894, -0.0011169, 0.0010837
9: -0.0054433, -0.0035165, -0.0052572, -0.0037250, -0.0009837, 0.0010138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005814, upper bound: 0.0004834
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005816, upper bound: 0.0004785
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037954, 0.0047574, 0.0038927, 0.0046410, -0.0005013, 0.0005035
1: 0.0018706, 0.0020096, 0.0018847, 0.0019928, -0.0000724, 0.0000727
2: 0.0117296, 0.0122615, 0.0117940, 0.0122077, -0.0002784, 0.0002771
3: -0.0025491, -0.0019991, -0.0024826, -0.0020547, -0.0002879, 0.0002866
4: -0.0018728, -0.0012774, -0.0018126, -0.0013494, -0.0003103, 0.0003117
5: 0.0053222, 0.0058857, 0.0053904, 0.0058287, -0.0002949, 0.0002936
6: -0.0011835, 0.0010524, -0.0009129, 0.0008262, -0.0011702, 0.0011651
7: -0.0039899, -0.0009449, -0.0036820, -0.0013134, -0.0015867, 0.0015938
8: 0.9864033, 0.9885483, 0.9866202, 0.9882887, -0.0011177, 0.0011227
9: -0.0054922, -0.0035451, -0.0052565, -0.0037420, -0.0010191, 0.0010146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005806, upper bound: 0.0004834
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005808, upper bound: 0.0004785
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037813, 0.0047333, 0.0038795, 0.0046569, -0.0005495, 0.0005237
1: 0.0018686, 0.0020061, 0.0018828, 0.0019951, -0.0000794, 0.0000757
2: 0.0117430, 0.0122693, 0.0117852, 0.0122150, -0.0002895, 0.0003038
3: -0.0025353, -0.0019910, -0.0024917, -0.0020471, -0.0002994, 0.0003142
4: -0.0018816, -0.0012923, -0.0018208, -0.0013396, -0.0003402, 0.0003242
5: 0.0053363, 0.0058939, 0.0053810, 0.0058365, -0.0003068, 0.0003219
6: -0.0011274, 0.0010851, -0.0009500, 0.0008570, -0.0012172, 0.0012773
7: -0.0040346, -0.0010212, -0.0037239, -0.0012629, -0.0017396, 0.0016577
8: 0.9863718, 0.9884945, 0.9865906, 0.9883243, -0.0012254, 0.0011677
9: -0.0054433, -0.0035165, -0.0052888, -0.0037152, -0.0010600, 0.0011123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005800, upper bound: 0.0004990
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0004902
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037954, 0.0047574, 0.0038871, 0.0046566, -0.0005498, 0.0005406
1: 0.0018706, 0.0020096, 0.0018839, 0.0019950, -0.0000794, 0.0000781
2: 0.0117296, 0.0122615, 0.0117854, 0.0122108, -0.0002989, 0.0003040
3: -0.0025491, -0.0019991, -0.0024915, -0.0020515, -0.0003091, 0.0003144
4: -0.0018728, -0.0012774, -0.0018161, -0.0013398, -0.0003403, 0.0003346
5: 0.0053222, 0.0058857, 0.0053812, 0.0058320, -0.0003167, 0.0003221
6: -0.0011835, 0.0010524, -0.0009492, 0.0008394, -0.0012564, 0.0012778
7: -0.0039899, -0.0009449, -0.0036999, -0.0012640, -0.0017403, 0.0017112
8: 0.9864033, 0.9885483, 0.9866076, 0.9883235, -0.0012259, 0.0012054
9: -0.0054922, -0.0035451, -0.0052881, -0.0037305, -0.0010942, 0.0011128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0004990
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0004902
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036967, 0.0047797, 0.0038843, 0.0046413, -0.0005990, 0.0005482
1: 0.0018564, 0.0020128, 0.0018835, 0.0019928, -0.0000865, 0.0000792
2: 0.0117173, 0.0123161, 0.0117938, 0.0122123, -0.0003031, 0.0003312
3: -0.0025619, -0.0019426, -0.0024827, -0.0020499, -0.0003135, 0.0003425
4: -0.0019340, -0.0012636, -0.0018178, -0.0013492, -0.0003708, 0.0003394
5: 0.0053091, 0.0059435, 0.0053902, 0.0058336, -0.0003211, 0.0003509
6: -0.0012354, 0.0012819, -0.0009137, 0.0008457, -0.0012742, 0.0013923
7: -0.0043026, -0.0008743, -0.0037085, -0.0013124, -0.0018962, 0.0017353
8: 0.9861830, 0.9885980, 0.9866015, 0.9882894, -0.0013357, 0.0012224
9: -0.0055373, -0.0033452, -0.0052572, -0.0037250, -0.0011096, 0.0012125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004712
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004680
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037089, 0.0048052, 0.0038927, 0.0046410, -0.0005997, 0.0005681
1: 0.0018581, 0.0020165, 0.0018847, 0.0019928, -0.0000866, 0.0000821
2: 0.0117032, 0.0123093, 0.0117940, 0.0122077, -0.0003141, 0.0003316
3: -0.0025765, -0.0019496, -0.0024826, -0.0020547, -0.0003248, 0.0003429
4: -0.0019264, -0.0012478, -0.0018126, -0.0013494, -0.0003712, 0.0003517
5: 0.0052942, 0.0059364, 0.0053904, 0.0058287, -0.0003328, 0.0003513
6: -0.0012946, 0.0012535, -0.0009129, 0.0008262, -0.0013204, 0.0013938
7: -0.0042639, -0.0007935, -0.0036820, -0.0013134, -0.0018983, 0.0017983
8: 0.9862103, 0.9886550, 0.9866202, 0.9882887, -0.0013372, 0.0012668
9: -0.0055890, -0.0033699, -0.0052565, -0.0037420, -0.0011499, 0.0012138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004712
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004680
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036967, 0.0047797, 0.0038795, 0.0046569, -0.0006477, 0.0005859
1: 0.0018564, 0.0020128, 0.0018828, 0.0019951, -0.0000936, 0.0000846
2: 0.0117173, 0.0123161, 0.0117852, 0.0122150, -0.0003239, 0.0003581
3: -0.0025619, -0.0019426, -0.0024917, -0.0020471, -0.0003350, 0.0003704
4: -0.0019340, -0.0012636, -0.0018208, -0.0013396, -0.0004009, 0.0003627
5: 0.0053091, 0.0059435, 0.0053810, 0.0058365, -0.0003432, 0.0003794
6: -0.0012354, 0.0012819, -0.0009500, 0.0008570, -0.0013617, 0.0015054
7: -0.0043026, -0.0008743, -0.0037239, -0.0012629, -0.0020503, 0.0018546
8: 0.9861830, 0.9885980, 0.9865906, 0.9883243, -0.0014443, 0.0013064
9: -0.0055373, -0.0033452, -0.0052888, -0.0037152, -0.0011859, 0.0013110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004768
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004718
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037089, 0.0048052, 0.0038871, 0.0046566, -0.0006482, 0.0006052
1: 0.0018581, 0.0020165, 0.0018839, 0.0019950, -0.0000936, 0.0000874
2: 0.0117032, 0.0123093, 0.0117854, 0.0122108, -0.0003346, 0.0003584
3: -0.0025765, -0.0019496, -0.0024915, -0.0020515, -0.0003461, 0.0003707
4: -0.0019264, -0.0012478, -0.0018161, -0.0013398, -0.0004013, 0.0003746
5: 0.0052942, 0.0059364, 0.0053812, 0.0058320, -0.0003545, 0.0003797
6: -0.0012946, 0.0012535, -0.0009492, 0.0008394, -0.0014066, 0.0015066
7: -0.0042639, -0.0007935, -0.0036999, -0.0012640, -0.0020519, 0.0019157
8: 0.9862103, 0.9886550, 0.9866076, 0.9883235, -0.0014454, 0.0013495
9: -0.0055890, -0.0033699, -0.0052881, -0.0037305, -0.0012250, 0.0013121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004768
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004717
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037813, 0.0047333, 0.0037719, 0.0047338, -0.0004088, 0.0004186
1: 0.0018686, 0.0020061, 0.0018672, 0.0020062, -0.0000591, 0.0000605
2: 0.0117430, 0.0122693, 0.0117427, 0.0122745, -0.0002314, 0.0002260
3: -0.0025353, -0.0019910, -0.0025356, -0.0019856, -0.0002394, 0.0002338
4: -0.0018816, -0.0012923, -0.0018874, -0.0012920, -0.0002531, 0.0002591
5: 0.0053363, 0.0058939, 0.0053360, 0.0058995, -0.0002452, 0.0002395
6: -0.0011274, 0.0010851, -0.0011286, 0.0011070, -0.0009730, 0.0009502
7: -0.0040346, -0.0010212, -0.0040643, -0.0010196, -0.0012941, 0.0013251
8: 0.9863718, 0.9884945, 0.9863509, 0.9884956, -0.0009116, 0.0009334
9: -0.0054433, -0.0035165, -0.0054444, -0.0034975, -0.0008473, 0.0008275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005814, upper bound: 0.0004834
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005816, upper bound: 0.0004785
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037954, 0.0047574, 0.0037810, 0.0047334, -0.0004100, 0.0004485
1: 0.0018706, 0.0020096, 0.0018685, 0.0020061, -0.0000592, 0.0000648
2: 0.0117296, 0.0122615, 0.0117429, 0.0122694, -0.0002479, 0.0002267
3: -0.0025491, -0.0019991, -0.0025354, -0.0019908, -0.0002564, 0.0002344
4: -0.0018728, -0.0012774, -0.0018818, -0.0012922, -0.0002538, 0.0002776
5: 0.0053222, 0.0058857, 0.0053362, 0.0058941, -0.0002627, 0.0002402
6: -0.0011835, 0.0010524, -0.0011279, 0.0010859, -0.0010423, 0.0009529
7: -0.0039899, -0.0009449, -0.0040356, -0.0010207, -0.0012978, 0.0014196
8: 0.9864033, 0.9885483, 0.9863712, 0.9884949, -0.0009142, 0.0010000
9: -0.0054922, -0.0035451, -0.0054437, -0.0035159, -0.0009077, 0.0008298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005806, upper bound: 0.0004834
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005808, upper bound: 0.0004786
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037813, 0.0047333, 0.0037689, 0.0047480, -0.0004562, 0.0004523
1: 0.0018686, 0.0020061, 0.0018668, 0.0020082, -0.0000659, 0.0000653
2: 0.0117430, 0.0122693, 0.0117348, 0.0122761, -0.0002501, 0.0002522
3: -0.0025353, -0.0019910, -0.0025437, -0.0019839, -0.0002586, 0.0002608
4: -0.0018816, -0.0012923, -0.0018893, -0.0012832, -0.0002824, 0.0002800
5: 0.0053363, 0.0058939, 0.0053277, 0.0059012, -0.0002650, 0.0002672
6: -0.0011274, 0.0010851, -0.0011616, 0.0011140, -0.0010513, 0.0010602
7: -0.0040346, -0.0010212, -0.0040739, -0.0009747, -0.0014439, 0.0014318
8: 0.9863718, 0.9884945, 0.9863442, 0.9885273, -0.0010171, 0.0010086
9: -0.0054433, -0.0035165, -0.0054731, -0.0034914, -0.0009155, 0.0009233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005800, upper bound: 0.0004990
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0004902
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037954, 0.0047574, 0.0037778, 0.0047476, -0.0004571, 0.0004827
1: 0.0018706, 0.0020096, 0.0018681, 0.0020082, -0.0000660, 0.0000697
2: 0.0117296, 0.0122615, 0.0117351, 0.0122712, -0.0002668, 0.0002527
3: -0.0025491, -0.0019991, -0.0025435, -0.0019890, -0.0002760, 0.0002614
4: -0.0018728, -0.0012774, -0.0018838, -0.0012835, -0.0002830, 0.0002988
5: 0.0053222, 0.0058857, 0.0053279, 0.0058960, -0.0002827, 0.0002678
6: -0.0011835, 0.0010524, -0.0011607, 0.0010933, -0.0011218, 0.0010625
7: -0.0039899, -0.0009449, -0.0040457, -0.0009760, -0.0014470, 0.0015278
8: 0.9864033, 0.9885483, 0.9863639, 0.9885263, -0.0010193, 0.0010762
9: -0.0054922, -0.0035451, -0.0054723, -0.0035094, -0.0009769, 0.0009253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0004990
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0004902
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036967, 0.0047797, 0.0037719, 0.0047338, -0.0005251, 0.0005000
1: 0.0018564, 0.0020128, 0.0018672, 0.0020062, -0.0000759, 0.0000722
2: 0.0117173, 0.0123161, 0.0117427, 0.0122745, -0.0002764, 0.0002903
3: -0.0025619, -0.0019426, -0.0025356, -0.0019856, -0.0002859, 0.0003003
4: -0.0019340, -0.0012636, -0.0018874, -0.0012920, -0.0003251, 0.0003095
5: 0.0053091, 0.0059435, 0.0053360, 0.0058995, -0.0002929, 0.0003076
6: -0.0012354, 0.0012819, -0.0011286, 0.0011070, -0.0011621, 0.0012206
7: -0.0043026, -0.0008743, -0.0040643, -0.0010196, -0.0016623, 0.0015827
8: 0.9861830, 0.9885980, 0.9863509, 0.9884956, -0.0011710, 0.0011149
9: -0.0055373, -0.0033452, -0.0054444, -0.0034975, -0.0010120, 0.0010629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004712
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004680
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037089, 0.0048052, 0.0037810, 0.0047334, -0.0005258, 0.0005259
1: 0.0018581, 0.0020165, 0.0018685, 0.0020061, -0.0000760, 0.0000760
2: 0.0117032, 0.0123093, 0.0117429, 0.0122694, -0.0002908, 0.0002907
3: -0.0025765, -0.0019496, -0.0025354, -0.0019908, -0.0003007, 0.0003006
4: -0.0019264, -0.0012478, -0.0018818, -0.0012922, -0.0003255, 0.0003256
5: 0.0052942, 0.0059364, 0.0053362, 0.0058941, -0.0003081, 0.0003080
6: -0.0012946, 0.0012535, -0.0011279, 0.0010859, -0.0012224, 0.0012220
7: -0.0042639, -0.0007935, -0.0040356, -0.0010207, -0.0016643, 0.0016649
8: 0.9862103, 0.9886550, 0.9863712, 0.9884949, -0.0011724, 0.0011728
9: -0.0055890, -0.0033699, -0.0054437, -0.0035159, -0.0010646, 0.0010642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004711
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004680
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036967, 0.0047797, 0.0037689, 0.0047480, -0.0005725, 0.0005337
1: 0.0018564, 0.0020128, 0.0018668, 0.0020082, -0.0000827, 0.0000771
2: 0.0117173, 0.0123161, 0.0117348, 0.0122761, -0.0002951, 0.0003165
3: -0.0025619, -0.0019426, -0.0025437, -0.0019839, -0.0003052, 0.0003274
4: -0.0019340, -0.0012636, -0.0018893, -0.0012832, -0.0003544, 0.0003303
5: 0.0053091, 0.0059435, 0.0053277, 0.0059012, -0.0003126, 0.0003354
6: -0.0012354, 0.0012819, -0.0011616, 0.0011140, -0.0012404, 0.0013306
7: -0.0043026, -0.0008743, -0.0040739, -0.0009747, -0.0018122, 0.0016893
8: 0.9861830, 0.9885980, 0.9863442, 0.9885273, -0.0012765, 0.0011900
9: -0.0055373, -0.0033452, -0.0054731, -0.0034914, -0.0010802, 0.0011588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004768
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004718
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037089, 0.0048052, 0.0037778, 0.0047476, -0.0005729, 0.0005601
1: 0.0018581, 0.0020165, 0.0018681, 0.0020082, -0.0000828, 0.0000809
2: 0.0117032, 0.0123093, 0.0117351, 0.0122712, -0.0003097, 0.0003167
3: -0.0025765, -0.0019496, -0.0025435, -0.0019890, -0.0003203, 0.0003276
4: -0.0019264, -0.0012478, -0.0018838, -0.0012835, -0.0003546, 0.0003467
5: 0.0052942, 0.0059364, 0.0053279, 0.0058960, -0.0003281, 0.0003356
6: -0.0012946, 0.0012535, -0.0011607, 0.0010933, -0.0013019, 0.0013316
7: -0.0042639, -0.0007935, -0.0040457, -0.0009760, -0.0018135, 0.0017731
8: 0.9862103, 0.9886550, 0.9863639, 0.9885263, -0.0012775, 0.0012490
9: -0.0055890, -0.0033699, -0.0054723, -0.0035094, -0.0011338, 0.0011596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004768
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004717
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037792, 0.0047474, 0.0038843, 0.0046413, -0.0005048, 0.0005021
1: 0.0018683, 0.0020082, 0.0018835, 0.0019928, -0.0000729, 0.0000725
2: 0.0117351, 0.0122704, 0.0117938, 0.0122123, -0.0002776, 0.0002791
3: -0.0025434, -0.0019898, -0.0024827, -0.0020499, -0.0002871, 0.0002886
4: -0.0018829, -0.0012835, -0.0018178, -0.0013492, -0.0003125, 0.0003108
5: 0.0053280, 0.0058952, 0.0053902, 0.0058336, -0.0002941, 0.0002957
6: -0.0011603, 0.0010901, -0.0009137, 0.0008457, -0.0011670, 0.0011733
7: -0.0040413, -0.0009764, -0.0037085, -0.0013124, -0.0015979, 0.0015894
8: 0.9863671, 0.9885260, 0.9866015, 0.9882894, -0.0011256, 0.0011196
9: -0.0054720, -0.0035122, -0.0052572, -0.0037250, -0.0010163, 0.0010217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004832
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004776
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037930, 0.0047711, 0.0038927, 0.0046410, -0.0005051, 0.0005240
1: 0.0018703, 0.0020116, 0.0018847, 0.0019928, -0.0000730, 0.0000757
2: 0.0117220, 0.0122628, 0.0117940, 0.0122077, -0.0002897, 0.0002792
3: -0.0025570, -0.0019977, -0.0024826, -0.0020547, -0.0002996, 0.0002888
4: -0.0018743, -0.0012689, -0.0018126, -0.0013494, -0.0003127, 0.0003243
5: 0.0053141, 0.0058871, 0.0053904, 0.0058287, -0.0003069, 0.0002959
6: -0.0012154, 0.0010580, -0.0009129, 0.0008262, -0.0012178, 0.0011740
7: -0.0039976, -0.0009014, -0.0036820, -0.0013134, -0.0015988, 0.0016586
8: 0.9863979, 0.9885790, 0.9866202, 0.9882887, -0.0011262, 0.0011683
9: -0.0055200, -0.0035402, -0.0052565, -0.0037420, -0.0010605, 0.0010223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004832
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004776
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037792, 0.0047474, 0.0038795, 0.0046569, -0.0005096, 0.0004953
1: 0.0018683, 0.0020082, 0.0018828, 0.0019951, -0.0000736, 0.0000716
2: 0.0117351, 0.0122704, 0.0117852, 0.0122150, -0.0002738, 0.0002818
3: -0.0025434, -0.0019898, -0.0024917, -0.0020471, -0.0002832, 0.0002914
4: -0.0018829, -0.0012835, -0.0018208, -0.0013396, -0.0003155, 0.0003066
5: 0.0053280, 0.0058952, 0.0053810, 0.0058365, -0.0002901, 0.0002986
6: -0.0011603, 0.0010901, -0.0009500, 0.0008570, -0.0011512, 0.0011846
7: -0.0040413, -0.0009764, -0.0037239, -0.0012629, -0.0016133, 0.0015678
8: 0.9863671, 0.9885260, 0.9865906, 0.9883243, -0.0011364, 0.0011044
9: -0.0054720, -0.0035122, -0.0052888, -0.0037152, -0.0010025, 0.0010316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004828
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004773
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037930, 0.0047711, 0.0038871, 0.0046566, -0.0005101, 0.0005128
1: 0.0018703, 0.0020116, 0.0018839, 0.0019950, -0.0000737, 0.0000741
2: 0.0117220, 0.0122628, 0.0117854, 0.0122108, -0.0002835, 0.0002820
3: -0.0025570, -0.0019977, -0.0024915, -0.0020515, -0.0002932, 0.0002917
4: -0.0018743, -0.0012689, -0.0018161, -0.0013398, -0.0003157, 0.0003174
5: 0.0053141, 0.0058871, 0.0053812, 0.0058320, -0.0003004, 0.0002988
6: -0.0012154, 0.0010580, -0.0009492, 0.0008394, -0.0011919, 0.0011855
7: -0.0039976, -0.0009014, -0.0036999, -0.0012640, -0.0016146, 0.0016233
8: 0.9863979, 0.9885790, 0.9866076, 0.9883235, -0.0011373, 0.0011435
9: -0.0055200, -0.0035402, -0.0052881, -0.0037305, -0.0010380, 0.0010324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004828
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004773
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036906, 0.0047980, 0.0038843, 0.0046413, -0.0005970, 0.0005595
1: 0.0018555, 0.0020155, 0.0018835, 0.0019928, -0.0000862, 0.0000808
2: 0.0117072, 0.0123194, 0.0117938, 0.0122123, -0.0003093, 0.0003301
3: -0.0025723, -0.0019392, -0.0024827, -0.0020499, -0.0003199, 0.0003414
4: -0.0019377, -0.0012523, -0.0018178, -0.0013492, -0.0003695, 0.0003464
5: 0.0052984, 0.0059471, 0.0053902, 0.0058336, -0.0003278, 0.0003497
6: -0.0012778, 0.0012959, -0.0009137, 0.0008457, -0.0013005, 0.0013876
7: -0.0043217, -0.0008164, -0.0037085, -0.0013124, -0.0018897, 0.0017712
8: 0.9861696, 0.9886388, 0.9866015, 0.9882894, -0.0013312, 0.0012477
9: -0.0055743, -0.0033330, -0.0052572, -0.0037250, -0.0011325, 0.0012084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004677
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037069, 0.0048189, 0.0038927, 0.0046410, -0.0005971, 0.0005838
1: 0.0018578, 0.0020185, 0.0018847, 0.0019928, -0.0000863, 0.0000843
2: 0.0116956, 0.0123104, 0.0117940, 0.0122077, -0.0003228, 0.0003301
3: -0.0025843, -0.0019484, -0.0024826, -0.0020547, -0.0003338, 0.0003414
4: -0.0019277, -0.0012393, -0.0018126, -0.0013494, -0.0003696, 0.0003614
5: 0.0052861, 0.0059376, 0.0053904, 0.0058287, -0.0003420, 0.0003498
6: -0.0013266, 0.0012582, -0.0009129, 0.0008262, -0.0013569, 0.0013879
7: -0.0042703, -0.0007500, -0.0036820, -0.0013134, -0.0018902, 0.0018480
8: 0.9862058, 0.9886855, 0.9866202, 0.9882887, -0.0013315, 0.0013018
9: -0.0056167, -0.0033658, -0.0052565, -0.0037420, -0.0011817, 0.0012086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004677
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036906, 0.0047980, 0.0038795, 0.0046569, -0.0006083, 0.0005576
1: 0.0018555, 0.0020155, 0.0018828, 0.0019951, -0.0000879, 0.0000806
2: 0.0117072, 0.0123194, 0.0117852, 0.0122150, -0.0003083, 0.0003363
3: -0.0025723, -0.0019392, -0.0024917, -0.0020471, -0.0003188, 0.0003478
4: -0.0019377, -0.0012523, -0.0018208, -0.0013396, -0.0003765, 0.0003452
5: 0.0052984, 0.0059471, 0.0053810, 0.0058365, -0.0003266, 0.0003563
6: -0.0012778, 0.0012959, -0.0009500, 0.0008570, -0.0012960, 0.0014138
7: -0.0043217, -0.0008164, -0.0037239, -0.0012629, -0.0019255, 0.0017650
8: 0.9861696, 0.9886388, 0.9865906, 0.9883243, -0.0013564, 0.0012433
9: -0.0055743, -0.0033330, -0.0052888, -0.0037152, -0.0011286, 0.0012312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004645
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037069, 0.0048189, 0.0038871, 0.0046566, -0.0006089, 0.0005773
1: 0.0018578, 0.0020185, 0.0018839, 0.0019950, -0.0000880, 0.0000834
2: 0.0116956, 0.0123104, 0.0117854, 0.0122108, -0.0003192, 0.0003366
3: -0.0025843, -0.0019484, -0.0024915, -0.0020515, -0.0003301, 0.0003482
4: -0.0019277, -0.0012393, -0.0018161, -0.0013398, -0.0003769, 0.0003573
5: 0.0052861, 0.0059376, 0.0053812, 0.0058320, -0.0003382, 0.0003567
6: -0.0013266, 0.0012582, -0.0009492, 0.0008394, -0.0013418, 0.0014152
7: -0.0042703, -0.0007500, -0.0036999, -0.0012640, -0.0019274, 0.0018274
8: 0.9862058, 0.9886855, 0.9866076, 0.9883235, -0.0013577, 0.0012872
9: -0.0056167, -0.0033658, -0.0052881, -0.0037305, -0.0011685, 0.0012324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004645
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037792, 0.0047474, 0.0037719, 0.0047338, -0.0004421, 0.0004657
1: 0.0018683, 0.0020082, 0.0018672, 0.0020062, -0.0000639, 0.0000673
2: 0.0117351, 0.0122704, 0.0117427, 0.0122745, -0.0002575, 0.0002444
3: -0.0025434, -0.0019898, -0.0025356, -0.0019856, -0.0002663, 0.0002528
4: -0.0018829, -0.0012835, -0.0018874, -0.0012920, -0.0002737, 0.0002883
5: 0.0053280, 0.0058952, 0.0053360, 0.0058995, -0.0002728, 0.0002590
6: -0.0011603, 0.0010901, -0.0011286, 0.0011070, -0.0010824, 0.0010276
7: -0.0040413, -0.0009764, -0.0040643, -0.0010196, -0.0013995, 0.0014742
8: 0.9863671, 0.9885260, 0.9863509, 0.9884956, -0.0009858, 0.0010384
9: -0.0054720, -0.0035122, -0.0054444, -0.0034975, -0.0009426, 0.0008948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004832
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004776
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037930, 0.0047711, 0.0037810, 0.0047334, -0.0004434, 0.0004913
1: 0.0018703, 0.0020116, 0.0018685, 0.0020061, -0.0000641, 0.0000710
2: 0.0117220, 0.0122628, 0.0117429, 0.0122694, -0.0002716, 0.0002452
3: -0.0025570, -0.0019977, -0.0025354, -0.0019908, -0.0002809, 0.0002536
4: -0.0018743, -0.0012689, -0.0018818, -0.0012922, -0.0002745, 0.0003041
5: 0.0053141, 0.0058871, 0.0053362, 0.0058941, -0.0002878, 0.0002598
6: -0.0012154, 0.0010580, -0.0011279, 0.0010859, -0.0011420, 0.0010307
7: -0.0039976, -0.0009014, -0.0040356, -0.0010207, -0.0014037, 0.0015553
8: 0.9863979, 0.9885790, 0.9863712, 0.9884949, -0.0009888, 0.0010956
9: -0.0055200, -0.0035402, -0.0054437, -0.0035159, -0.0009945, 0.0008976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004832
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004776
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037792, 0.0047474, 0.0037689, 0.0047480, -0.0004150, 0.0004249
1: 0.0018683, 0.0020082, 0.0018668, 0.0020082, -0.0000600, 0.0000614
2: 0.0117351, 0.0122704, 0.0117348, 0.0122761, -0.0002349, 0.0002294
3: -0.0025434, -0.0019898, -0.0025437, -0.0019839, -0.0002429, 0.0002373
4: -0.0018829, -0.0012835, -0.0018893, -0.0012832, -0.0002569, 0.0002630
5: 0.0053280, 0.0058952, 0.0053277, 0.0059012, -0.0002489, 0.0002431
6: -0.0011603, 0.0010901, -0.0011616, 0.0011140, -0.0009875, 0.0009646
7: -0.0040413, -0.0009764, -0.0040739, -0.0009747, -0.0013137, 0.0013449
8: 0.9863671, 0.9885260, 0.9863442, 0.9885273, -0.0009254, 0.0009474
9: -0.0054720, -0.0035122, -0.0054731, -0.0034914, -0.0008600, 0.0008400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004828
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004773
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037930, 0.0047711, 0.0037778, 0.0047476, -0.0004149, 0.0004543
1: 0.0018703, 0.0020116, 0.0018681, 0.0020082, -0.0000599, 0.0000656
2: 0.0117220, 0.0122628, 0.0117351, 0.0122712, -0.0002511, 0.0002294
3: -0.0025570, -0.0019977, -0.0025435, -0.0019890, -0.0002597, 0.0002373
4: -0.0018743, -0.0012689, -0.0018838, -0.0012835, -0.0002568, 0.0002812
5: 0.0053141, 0.0058871, 0.0053279, 0.0058960, -0.0002661, 0.0002431
6: -0.0012154, 0.0010580, -0.0011607, 0.0010933, -0.0010558, 0.0009644
7: -0.0039976, -0.0009014, -0.0040457, -0.0009760, -0.0013134, 0.0014379
8: 0.9863979, 0.9885790, 0.9863639, 0.9885263, -0.0009252, 0.0010129
9: -0.0055200, -0.0035402, -0.0054723, -0.0035094, -0.0009195, 0.0008398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004828
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004773
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036906, 0.0047980, 0.0037719, 0.0047338, -0.0005499, 0.0005370
1: 0.0018555, 0.0020155, 0.0018672, 0.0020062, -0.0000794, 0.0000776
2: 0.0117072, 0.0123194, 0.0117427, 0.0122745, -0.0002969, 0.0003040
3: -0.0025723, -0.0019392, -0.0025356, -0.0019856, -0.0003070, 0.0003144
4: -0.0019377, -0.0012523, -0.0018874, -0.0012920, -0.0003404, 0.0003324
5: 0.0052984, 0.0059471, 0.0053360, 0.0058995, -0.0003146, 0.0003221
6: -0.0012778, 0.0012959, -0.0011286, 0.0011070, -0.0012481, 0.0012781
7: -0.0043217, -0.0008164, -0.0040643, -0.0010196, -0.0017407, 0.0016997
8: 0.9861696, 0.9886388, 0.9863509, 0.9884956, -0.0012262, 0.0011973
9: -0.0055743, -0.0033330, -0.0054444, -0.0034975, -0.0010869, 0.0011130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004677
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037069, 0.0048189, 0.0037810, 0.0047334, -0.0005507, 0.0005575
1: 0.0018578, 0.0020185, 0.0018685, 0.0020061, -0.0000796, 0.0000805
2: 0.0116956, 0.0123104, 0.0117429, 0.0122694, -0.0003082, 0.0003045
3: -0.0025843, -0.0019484, -0.0025354, -0.0019908, -0.0003188, 0.0003149
4: -0.0019277, -0.0012393, -0.0018818, -0.0012922, -0.0003409, 0.0003451
5: 0.0052861, 0.0059376, 0.0053362, 0.0058941, -0.0003266, 0.0003226
6: -0.0013266, 0.0012582, -0.0011279, 0.0010859, -0.0012957, 0.0012800
7: -0.0042703, -0.0007500, -0.0040356, -0.0010207, -0.0017432, 0.0017646
8: 0.9862058, 0.9886855, 0.9863712, 0.9884949, -0.0012279, 0.0012430
9: -0.0056167, -0.0033658, -0.0054437, -0.0035159, -0.0011283, 0.0011146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004677
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036906, 0.0047980, 0.0037689, 0.0047480, -0.0005316, 0.0005060
1: 0.0018555, 0.0020155, 0.0018668, 0.0020082, -0.0000768, 0.0000731
2: 0.0117072, 0.0123194, 0.0117348, 0.0122761, -0.0002798, 0.0002939
3: -0.0025723, -0.0019392, -0.0025437, -0.0019839, -0.0002893, 0.0003040
4: -0.0019377, -0.0012523, -0.0018893, -0.0012832, -0.0003291, 0.0003132
5: 0.0052984, 0.0059471, 0.0053277, 0.0059012, -0.0002964, 0.0003114
6: -0.0012778, 0.0012959, -0.0011616, 0.0011140, -0.0011761, 0.0012355
7: -0.0043217, -0.0008164, -0.0040739, -0.0009747, -0.0016827, 0.0016018
8: 0.9861696, 0.9886388, 0.9863442, 0.9885273, -0.0011853, 0.0011283
9: -0.0055743, -0.0033330, -0.0054731, -0.0034914, -0.0010242, 0.0010760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004645
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037069, 0.0048189, 0.0037778, 0.0047476, -0.0005309, 0.0005317
1: 0.0018578, 0.0020185, 0.0018681, 0.0020082, -0.0000767, 0.0000768
2: 0.0116956, 0.0123104, 0.0117351, 0.0122712, -0.0002939, 0.0002935
3: -0.0025843, -0.0019484, -0.0025435, -0.0019890, -0.0003040, 0.0003036
4: -0.0019277, -0.0012393, -0.0018838, -0.0012835, -0.0003286, 0.0003291
5: 0.0052861, 0.0059376, 0.0053279, 0.0058960, -0.0003114, 0.0003110
6: -0.0013266, 0.0012582, -0.0011607, 0.0010933, -0.0012357, 0.0012339
7: -0.0042703, -0.0007500, -0.0040457, -0.0009760, -0.0016804, 0.0016829
8: 0.9862058, 0.9886855, 0.9863639, 0.9885263, -0.0011837, 0.0011855
9: -0.0056167, -0.0033658, -0.0054723, -0.0035094, -0.0010761, 0.0010745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004645
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037813, 0.0047333, 0.0037953, 0.0046916, -0.0005793, 0.0005997
1: 0.0018686, 0.0020061, 0.0018706, 0.0020001, -0.0000837, 0.0000866
2: 0.0117430, 0.0122693, 0.0117660, 0.0122615, -0.0003316, 0.0003203
3: -0.0025353, -0.0019910, -0.0025115, -0.0019990, -0.0003429, 0.0003312
4: -0.0018816, -0.0012923, -0.0018729, -0.0013181, -0.0003586, 0.0003712
5: 0.0053363, 0.0058939, 0.0053607, 0.0058858, -0.0003513, 0.0003393
6: -0.0011274, 0.0010851, -0.0010306, 0.0010527, -0.0013939, 0.0013464
7: -0.0040346, -0.0010212, -0.0039904, -0.0011531, -0.0018337, 0.0018983
8: 0.9863718, 0.9884945, 0.9864029, 0.9884016, -0.0012917, 0.0013372
9: -0.0054433, -0.0035165, -0.0053590, -0.0035448, -0.0012139, 0.0011725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005697, upper bound: 0.0005328
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005711, upper bound: 0.0005277
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037954, 0.0047574, 0.0038038, 0.0046913, -0.0005796, 0.0006156
1: 0.0018706, 0.0020096, 0.0018718, 0.0020001, -0.0000837, 0.0000889
2: 0.0117296, 0.0122615, 0.0117661, 0.0122568, -0.0003403, 0.0003204
3: -0.0025491, -0.0019991, -0.0025114, -0.0020039, -0.0003520, 0.0003314
4: -0.0018728, -0.0012774, -0.0018676, -0.0013183, -0.0003588, 0.0003810
5: 0.0053222, 0.0058857, 0.0053609, 0.0058808, -0.0003606, 0.0003395
6: -0.0011835, 0.0010524, -0.0010300, 0.0010328, -0.0014307, 0.0013471
7: -0.0039899, -0.0009449, -0.0039633, -0.0011540, -0.0018347, 0.0019485
8: 0.9864033, 0.9885483, 0.9864220, 0.9884010, -0.0012924, 0.0013726
9: -0.0054922, -0.0035451, -0.0053585, -0.0035621, -0.0012459, 0.0011731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005684, upper bound: 0.0005328
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005703, upper bound: 0.0005277
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037813, 0.0047333, 0.0037915, 0.0047058, -0.0006173, 0.0006263
1: 0.0018686, 0.0020061, 0.0018701, 0.0020022, -0.0000892, 0.0000905
2: 0.0117430, 0.0122693, 0.0117581, 0.0122636, -0.0003463, 0.0003413
3: -0.0025353, -0.0019910, -0.0025196, -0.0019968, -0.0003581, 0.0003530
4: -0.0018816, -0.0012923, -0.0018753, -0.0013093, -0.0003821, 0.0003877
5: 0.0053363, 0.0058939, 0.0053524, 0.0058880, -0.0003669, 0.0003616
6: -0.0011274, 0.0010851, -0.0010636, 0.0010615, -0.0014558, 0.0014347
7: -0.0040346, -0.0010212, -0.0040024, -0.0011082, -0.0019540, 0.0019826
8: 0.9863718, 0.9884945, 0.9863945, 0.9884333, -0.0013764, 0.0013966
9: -0.0054433, -0.0035165, -0.0053877, -0.0035371, -0.0012678, 0.0012494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005689, upper bound: 0.0005495
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005699, upper bound: 0.0005424
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037954, 0.0047574, 0.0038000, 0.0047055, -0.0006175, 0.0006419
1: 0.0018706, 0.0020096, 0.0018713, 0.0020021, -0.0000892, 0.0000927
2: 0.0117296, 0.0122615, 0.0117583, 0.0122589, -0.0003549, 0.0003414
3: -0.0025491, -0.0019991, -0.0025195, -0.0020017, -0.0003670, 0.0003531
4: -0.0018728, -0.0012774, -0.0018700, -0.0013095, -0.0003822, 0.0003973
5: 0.0053222, 0.0058857, 0.0053526, 0.0058830, -0.0003760, 0.0003617
6: -0.0011835, 0.0010524, -0.0010629, 0.0010418, -0.0014919, 0.0014352
7: -0.0039899, -0.0009449, -0.0039755, -0.0011091, -0.0019547, 0.0020319
8: 0.9864033, 0.9885483, 0.9864134, 0.9884326, -0.0013769, 0.0014313
9: -0.0054922, -0.0035451, -0.0053872, -0.0035543, -0.0012992, 0.0012499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005675, upper bound: 0.0005495
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005691, upper bound: 0.0005424
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036967, 0.0047797, 0.0037953, 0.0046916, -0.0005271, 0.0005119
1: 0.0018564, 0.0020128, 0.0018706, 0.0020001, -0.0000761, 0.0000740
2: 0.0117173, 0.0123161, 0.0117660, 0.0122615, -0.0002830, 0.0002914
3: -0.0025619, -0.0019426, -0.0025115, -0.0019990, -0.0002927, 0.0003014
4: -0.0019340, -0.0012636, -0.0018729, -0.0013181, -0.0003263, 0.0003169
5: 0.0053091, 0.0059435, 0.0053607, 0.0058858, -0.0002999, 0.0003088
6: -0.0012354, 0.0012819, -0.0010306, 0.0010527, -0.0011898, 0.0012251
7: -0.0043026, -0.0008743, -0.0039904, -0.0011531, -0.0016685, 0.0016204
8: 0.9861830, 0.9885980, 0.9864029, 0.9884016, -0.0011753, 0.0011414
9: -0.0055373, -0.0033452, -0.0053590, -0.0035448, -0.0010361, 0.0010669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004715
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004686
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037089, 0.0048052, 0.0038038, 0.0046913, -0.0005276, 0.0005300
1: 0.0018581, 0.0020165, 0.0018718, 0.0020001, -0.0000762, 0.0000766
2: 0.0117032, 0.0123093, 0.0117661, 0.0122568, -0.0002930, 0.0002917
3: -0.0025765, -0.0019496, -0.0025114, -0.0020039, -0.0003030, 0.0003017
4: -0.0019264, -0.0012478, -0.0018676, -0.0013183, -0.0003266, 0.0003281
5: 0.0052942, 0.0059364, 0.0053609, 0.0058808, -0.0003105, 0.0003091
6: -0.0012946, 0.0012535, -0.0010300, 0.0010328, -0.0012318, 0.0012263
7: -0.0042639, -0.0007935, -0.0039633, -0.0011540, -0.0016701, 0.0016776
8: 0.9862103, 0.9886550, 0.9864220, 0.9884010, -0.0011765, 0.0011817
9: -0.0055890, -0.0033699, -0.0053585, -0.0035621, -0.0010727, 0.0010679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004715
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004686
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036967, 0.0047797, 0.0037915, 0.0047058, -0.0005748, 0.0005490
1: 0.0018564, 0.0020128, 0.0018701, 0.0020022, -0.0000830, 0.0000793
2: 0.0117173, 0.0123161, 0.0117581, 0.0122636, -0.0003036, 0.0003178
3: -0.0025619, -0.0019426, -0.0025196, -0.0019968, -0.0003140, 0.0003287
4: -0.0019340, -0.0012636, -0.0018753, -0.0013093, -0.0003558, 0.0003399
5: 0.0053091, 0.0059435, 0.0053524, 0.0058880, -0.0003216, 0.0003367
6: -0.0012354, 0.0012819, -0.0010636, 0.0010615, -0.0012761, 0.0013361
7: -0.0043026, -0.0008743, -0.0040024, -0.0011082, -0.0018197, 0.0017380
8: 0.9861830, 0.9885980, 0.9863945, 0.9884333, -0.0012818, 0.0012243
9: -0.0055373, -0.0033452, -0.0053877, -0.0035371, -0.0011113, 0.0011635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004768
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004725
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037089, 0.0048052, 0.0038000, 0.0047055, -0.0005752, 0.0005666
1: 0.0018581, 0.0020165, 0.0018713, 0.0020021, -0.0000831, 0.0000819
2: 0.0117032, 0.0123093, 0.0117583, 0.0122589, -0.0003133, 0.0003180
3: -0.0025765, -0.0019496, -0.0025195, -0.0020017, -0.0003240, 0.0003289
4: -0.0019264, -0.0012478, -0.0018700, -0.0013095, -0.0003561, 0.0003507
5: 0.0052942, 0.0059364, 0.0053526, 0.0058830, -0.0003319, 0.0003370
6: -0.0012946, 0.0012535, -0.0010629, 0.0010418, -0.0013169, 0.0013370
7: -0.0042639, -0.0007935, -0.0039755, -0.0011091, -0.0018209, 0.0017935
8: 0.9862103, 0.9886550, 0.9864134, 0.9884326, -0.0012827, 0.0012634
9: -0.0055890, -0.0033699, -0.0053872, -0.0035543, -0.0011468, 0.0011643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004768
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004725
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037813, 0.0047333, 0.0036864, 0.0047801, -0.0004902, 0.0005334
1: 0.0018686, 0.0020061, 0.0018549, 0.0020129, -0.0000708, 0.0000771
2: 0.0117430, 0.0122693, 0.0117171, 0.0123217, -0.0002949, 0.0002710
3: -0.0025353, -0.0019910, -0.0025621, -0.0019367, -0.0003050, 0.0002803
4: -0.0018816, -0.0012923, -0.0019403, -0.0012633, -0.0003035, 0.0003302
5: 0.0053363, 0.0058939, 0.0053089, 0.0059495, -0.0003125, 0.0002872
6: -0.0011274, 0.0010851, -0.0012363, 0.0013057, -0.0012397, 0.0011395
7: -0.0040346, -0.0010212, -0.0043350, -0.0008730, -0.0015518, 0.0016884
8: 0.9863718, 0.9884945, 0.9861602, 0.9885989, -0.0010931, 0.0011893
9: -0.0054433, -0.0035165, -0.0055381, -0.0033244, -0.0010796, 0.0009923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005697, upper bound: 0.0005328
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005711, upper bound: 0.0005277
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037954, 0.0047574, 0.0036952, 0.0047799, -0.0004914, 0.0005610
1: 0.0018706, 0.0020096, 0.0018562, 0.0020129, -0.0000710, 0.0000810
2: 0.0117296, 0.0122615, 0.0117172, 0.0123169, -0.0003102, 0.0002717
3: -0.0025491, -0.0019991, -0.0025620, -0.0019418, -0.0003208, 0.0002810
4: -0.0018728, -0.0012774, -0.0019349, -0.0012635, -0.0003042, 0.0003473
5: 0.0053222, 0.0058857, 0.0053090, 0.0059444, -0.0003286, 0.0002878
6: -0.0011835, 0.0010524, -0.0012357, 0.0012853, -0.0013039, 0.0011420
7: -0.0039899, -0.0009449, -0.0043072, -0.0008738, -0.0015554, 0.0017758
8: 0.9864033, 0.9885483, 0.9861797, 0.9885983, -0.0010956, 0.0012509
9: -0.0054922, -0.0035451, -0.0055376, -0.0033422, -0.0011355, 0.0009945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005684, upper bound: 0.0005328
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005703, upper bound: 0.0005277
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037813, 0.0047333, 0.0036808, 0.0047985, -0.0005274, 0.0005591
1: 0.0018686, 0.0020061, 0.0018541, 0.0020155, -0.0000762, 0.0000808
2: 0.0117430, 0.0122693, 0.0117069, 0.0123249, -0.0003091, 0.0002916
3: -0.0025353, -0.0019910, -0.0025726, -0.0019335, -0.0003197, 0.0003016
4: -0.0018816, -0.0012923, -0.0019438, -0.0012520, -0.0003265, 0.0003461
5: 0.0053363, 0.0058939, 0.0052981, 0.0059529, -0.0003275, 0.0003090
6: -0.0011274, 0.0010851, -0.0012790, 0.0013189, -0.0012996, 0.0012258
7: -0.0040346, -0.0010212, -0.0043529, -0.0008149, -0.0016695, 0.0017699
8: 0.9863718, 0.9884945, 0.9861476, 0.9886399, -0.0011760, 0.0012468
9: -0.0054433, -0.0035165, -0.0055753, -0.0033130, -0.0011317, 0.0010675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005689, upper bound: 0.0005495
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005699, upper bound: 0.0005424
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037954, 0.0047574, 0.0036904, 0.0047981, -0.0005284, 0.0005876
1: 0.0018706, 0.0020096, 0.0018555, 0.0020155, -0.0000763, 0.0000849
2: 0.0117296, 0.0122615, 0.0117071, 0.0123195, -0.0003249, 0.0002921
3: -0.0025491, -0.0019991, -0.0025724, -0.0019390, -0.0003360, 0.0003021
4: -0.0018728, -0.0012774, -0.0019378, -0.0012522, -0.0003271, 0.0003637
5: 0.0053222, 0.0058857, 0.0052983, 0.0059472, -0.0003442, 0.0003095
6: -0.0011835, 0.0010524, -0.0012781, 0.0012964, -0.0013657, 0.0012281
7: -0.0039899, -0.0009449, -0.0043223, -0.0008160, -0.0016725, 0.0018600
8: 0.9864033, 0.9885483, 0.9861692, 0.9886391, -0.0011782, 0.0013102
9: -0.0054922, -0.0035451, -0.0055746, -0.0033326, -0.0011893, 0.0010695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005675, upper bound: 0.0005495
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005691, upper bound: 0.0005424
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036967, 0.0047797, 0.0036864, 0.0047801, -0.0004356, 0.0004456
1: 0.0018564, 0.0020128, 0.0018549, 0.0020129, -0.0000629, 0.0000644
2: 0.0117173, 0.0123161, 0.0117171, 0.0123217, -0.0002464, 0.0002409
3: -0.0025619, -0.0019426, -0.0025621, -0.0019367, -0.0002548, 0.0002491
4: -0.0019340, -0.0012636, -0.0019403, -0.0012633, -0.0002697, 0.0002758
5: 0.0053091, 0.0059435, 0.0053089, 0.0059495, -0.0002610, 0.0002552
6: -0.0012354, 0.0012819, -0.0012363, 0.0013057, -0.0010357, 0.0010125
7: -0.0043026, -0.0008743, -0.0043350, -0.0008730, -0.0013790, 0.0014105
8: 0.9861830, 0.9885980, 0.9861602, 0.9885989, -0.0009714, 0.0009936
9: -0.0055373, -0.0033452, -0.0055381, -0.0033244, -0.0009019, 0.0008818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004715
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004686
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037089, 0.0048052, 0.0036952, 0.0047799, -0.0004372, 0.0004757
1: 0.0018581, 0.0020165, 0.0018562, 0.0020129, -0.0000632, 0.0000687
2: 0.0117032, 0.0123093, 0.0117172, 0.0123169, -0.0002630, 0.0002417
3: -0.0025765, -0.0019496, -0.0025620, -0.0019418, -0.0002720, 0.0002500
4: -0.0019264, -0.0012478, -0.0019349, -0.0012635, -0.0002706, 0.0002945
5: 0.0052942, 0.0059364, 0.0053090, 0.0059444, -0.0002787, 0.0002561
6: -0.0012946, 0.0012535, -0.0012357, 0.0012853, -0.0011057, 0.0010161
7: -0.0042639, -0.0007935, -0.0043072, -0.0008738, -0.0013838, 0.0015058
8: 0.9862103, 0.9886550, 0.9861797, 0.9885983, -0.0009748, 0.0010607
9: -0.0055890, -0.0033699, -0.0055376, -0.0033422, -0.0009629, 0.0008848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004715
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004686
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036967, 0.0047797, 0.0036808, 0.0047985, -0.0004820, 0.0004790
1: 0.0018564, 0.0020128, 0.0018541, 0.0020155, -0.0000696, 0.0000692
2: 0.0117173, 0.0123161, 0.0117069, 0.0123249, -0.0002648, 0.0002665
3: -0.0025619, -0.0019426, -0.0025726, -0.0019335, -0.0002739, 0.0002756
4: -0.0019340, -0.0012636, -0.0019438, -0.0012520, -0.0002984, 0.0002965
5: 0.0053091, 0.0059435, 0.0052981, 0.0059529, -0.0002806, 0.0002824
6: -0.0012354, 0.0012819, -0.0012790, 0.0013189, -0.0011133, 0.0011203
7: -0.0043026, -0.0008743, -0.0043529, -0.0008149, -0.0015257, 0.0015162
8: 0.9861830, 0.9885980, 0.9861476, 0.9886399, -0.0010748, 0.0010681
9: -0.0055373, -0.0033452, -0.0055753, -0.0033130, -0.0009695, 0.0009756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004768
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004725
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037089, 0.0048052, 0.0036904, 0.0047981, -0.0004833, 0.0005096
1: 0.0018581, 0.0020165, 0.0018555, 0.0020155, -0.0000698, 0.0000736
2: 0.0117032, 0.0123093, 0.0117071, 0.0123195, -0.0002818, 0.0002672
3: -0.0025765, -0.0019496, -0.0025724, -0.0019390, -0.0002914, 0.0002764
4: -0.0019264, -0.0012478, -0.0019378, -0.0012522, -0.0002992, 0.0003155
5: 0.0052942, 0.0059364, 0.0052983, 0.0059472, -0.0002985, 0.0002831
6: -0.0012946, 0.0012535, -0.0012781, 0.0012964, -0.0011845, 0.0011233
7: -0.0042639, -0.0007935, -0.0043223, -0.0008160, -0.0015299, 0.0016132
8: 0.9862103, 0.9886550, 0.9861692, 0.9886391, -0.0010777, 0.0011364
9: -0.0055890, -0.0033699, -0.0055746, -0.0033326, -0.0010315, 0.0009782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004768
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004725
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037792, 0.0047474, 0.0037953, 0.0046916, -0.0005832, 0.0006158
1: 0.0018683, 0.0020082, 0.0018706, 0.0020001, -0.0000843, 0.0000890
2: 0.0117351, 0.0122704, 0.0117660, 0.0122615, -0.0003405, 0.0003224
3: -0.0025434, -0.0019898, -0.0025115, -0.0019990, -0.0003521, 0.0003335
4: -0.0018829, -0.0012835, -0.0018729, -0.0013181, -0.0003610, 0.0003812
5: 0.0053280, 0.0058952, 0.0053607, 0.0058858, -0.0003607, 0.0003416
6: -0.0011603, 0.0010901, -0.0010306, 0.0010527, -0.0014313, 0.0013555
7: -0.0040413, -0.0009764, -0.0039904, -0.0011531, -0.0018461, 0.0019492
8: 0.9863671, 0.9885260, 0.9864029, 0.9884016, -0.0013004, 0.0013731
9: -0.0054720, -0.0035122, -0.0053590, -0.0035448, -0.0012464, 0.0011804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005327
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005275
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037930, 0.0047711, 0.0038038, 0.0046913, -0.0005834, 0.0006360
1: 0.0018703, 0.0020116, 0.0018718, 0.0020001, -0.0000843, 0.0000919
2: 0.0117220, 0.0122628, 0.0117661, 0.0122568, -0.0003516, 0.0003226
3: -0.0025570, -0.0019977, -0.0025114, -0.0020039, -0.0003637, 0.0003336
4: -0.0018743, -0.0012689, -0.0018676, -0.0013183, -0.0003611, 0.0003937
5: 0.0053141, 0.0058871, 0.0053609, 0.0058808, -0.0003726, 0.0003418
6: -0.0012154, 0.0010580, -0.0010300, 0.0010328, -0.0014783, 0.0013560
7: -0.0039976, -0.0009014, -0.0039633, -0.0011540, -0.0018468, 0.0020133
8: 0.9863979, 0.9885790, 0.9864220, 0.9884010, -0.0013009, 0.0014182
9: -0.0055200, -0.0035402, -0.0053585, -0.0035621, -0.0012874, 0.0011809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005651, upper bound: 0.0005327
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005654, upper bound: 0.0005275
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037792, 0.0047474, 0.0037915, 0.0047058, -0.0005881, 0.0006082
1: 0.0018683, 0.0020082, 0.0018701, 0.0020022, -0.0000850, 0.0000879
2: 0.0117351, 0.0122704, 0.0117581, 0.0122636, -0.0003362, 0.0003252
3: -0.0025434, -0.0019898, -0.0025196, -0.0019968, -0.0003478, 0.0003363
4: -0.0018829, -0.0012835, -0.0018753, -0.0013093, -0.0003641, 0.0003765
5: 0.0053280, 0.0058952, 0.0053524, 0.0058880, -0.0003563, 0.0003445
6: -0.0011603, 0.0010901, -0.0010636, 0.0010615, -0.0014136, 0.0013670
7: -0.0040413, -0.0009764, -0.0040024, -0.0011082, -0.0018617, 0.0019252
8: 0.9863671, 0.9885260, 0.9863945, 0.9884333, -0.0013115, 0.0013561
9: -0.0054720, -0.0035122, -0.0053877, -0.0035371, -0.0012310, 0.0011905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005327
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005275
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037930, 0.0047711, 0.0038000, 0.0047055, -0.0005885, 0.0006239
1: 0.0018703, 0.0020116, 0.0018713, 0.0020021, -0.0000850, 0.0000901
2: 0.0117220, 0.0122628, 0.0117583, 0.0122589, -0.0003449, 0.0003254
3: -0.0025570, -0.0019977, -0.0025195, -0.0020017, -0.0003567, 0.0003365
4: -0.0018743, -0.0012689, -0.0018700, -0.0013095, -0.0003643, 0.0003862
5: 0.0053141, 0.0058871, 0.0053526, 0.0058830, -0.0003655, 0.0003447
6: -0.0012154, 0.0010580, -0.0010629, 0.0010418, -0.0014501, 0.0013678
7: -0.0039976, -0.0009014, -0.0039755, -0.0011091, -0.0018628, 0.0019749
8: 0.9863979, 0.9885790, 0.9864134, 0.9884326, -0.0013122, 0.0013912
9: -0.0055200, -0.0035402, -0.0053872, -0.0035543, -0.0012628, 0.0011911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005651, upper bound: 0.0005327
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005654, upper bound: 0.0005275
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036906, 0.0047980, 0.0037953, 0.0046916, -0.0005311, 0.0005274
1: 0.0018555, 0.0020155, 0.0018706, 0.0020001, -0.0000767, 0.0000762
2: 0.0117072, 0.0123194, 0.0117660, 0.0122615, -0.0002916, 0.0002936
3: -0.0025723, -0.0019392, -0.0025115, -0.0019990, -0.0003016, 0.0003037
4: -0.0019377, -0.0012523, -0.0018729, -0.0013181, -0.0003288, 0.0003265
5: 0.0052984, 0.0059471, 0.0053607, 0.0058858, -0.0003090, 0.0003111
6: -0.0012778, 0.0012959, -0.0010306, 0.0010527, -0.0012258, 0.0012345
7: -0.0043217, -0.0008164, -0.0039904, -0.0011531, -0.0016812, 0.0016695
8: 0.9861696, 0.9886388, 0.9864029, 0.9884016, -0.0011843, 0.0011760
9: -0.0055743, -0.0033330, -0.0053590, -0.0035448, -0.0010675, 0.0010750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004685
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037069, 0.0048189, 0.0038038, 0.0046913, -0.0005316, 0.0005499
1: 0.0018578, 0.0020185, 0.0018718, 0.0020001, -0.0000768, 0.0000794
2: 0.0116956, 0.0123104, 0.0117661, 0.0122568, -0.0003040, 0.0002939
3: -0.0025843, -0.0019484, -0.0025114, -0.0020039, -0.0003144, 0.0003040
4: -0.0019277, -0.0012393, -0.0018676, -0.0013183, -0.0003291, 0.0003404
5: 0.0052861, 0.0059376, 0.0053609, 0.0058808, -0.0003221, 0.0003114
6: -0.0013266, 0.0012582, -0.0010300, 0.0010328, -0.0012781, 0.0012355
7: -0.0042703, -0.0007500, -0.0039633, -0.0011540, -0.0016827, 0.0017407
8: 0.9862058, 0.9886855, 0.9864220, 0.9884010, -0.0011853, 0.0012262
9: -0.0056167, -0.0033658, -0.0053585, -0.0035621, -0.0011131, 0.0010760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004685
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036906, 0.0047980, 0.0037915, 0.0047058, -0.0005363, 0.0005214
1: 0.0018555, 0.0020155, 0.0018701, 0.0020022, -0.0000775, 0.0000753
2: 0.0117072, 0.0123194, 0.0117581, 0.0122636, -0.0002883, 0.0002965
3: -0.0025723, -0.0019392, -0.0025196, -0.0019968, -0.0002981, 0.0003066
4: -0.0019377, -0.0012523, -0.0018753, -0.0013093, -0.0003320, 0.0003228
5: 0.0052984, 0.0059471, 0.0053524, 0.0058880, -0.0003054, 0.0003142
6: -0.0012778, 0.0012959, -0.0010636, 0.0010615, -0.0012119, 0.0012465
7: -0.0043217, -0.0008164, -0.0040024, -0.0011082, -0.0016976, 0.0016505
8: 0.9861696, 0.9886388, 0.9863945, 0.9884333, -0.0011958, 0.0011627
9: -0.0055743, -0.0033330, -0.0053877, -0.0035371, -0.0010554, 0.0010855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004682
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004654
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037069, 0.0048189, 0.0038000, 0.0047055, -0.0005361, 0.0005395
1: 0.0018578, 0.0020185, 0.0018713, 0.0020021, -0.0000774, 0.0000779
2: 0.0116956, 0.0123104, 0.0117583, 0.0122589, -0.0002983, 0.0002964
3: -0.0025843, -0.0019484, -0.0025195, -0.0020017, -0.0003085, 0.0003065
4: -0.0019277, -0.0012393, -0.0018700, -0.0013095, -0.0003318, 0.0003340
5: 0.0052861, 0.0059376, 0.0053526, 0.0058830, -0.0003160, 0.0003140
6: -0.0013266, 0.0012582, -0.0010629, 0.0010418, -0.0012540, 0.0012460
7: -0.0042703, -0.0007500, -0.0039755, -0.0011091, -0.0016970, 0.0017078
8: 0.9862058, 0.9886855, 0.9864134, 0.9884326, -0.0011954, 0.0012030
9: -0.0056167, -0.0033658, -0.0053872, -0.0035543, -0.0010920, 0.0010851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004683
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004654
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0037792, 0.0047474, 0.0036864, 0.0047801, -0.0005235, 0.0005805
1: 0.0018683, 0.0020082, 0.0018549, 0.0020129, -0.0000756, 0.0000839
2: 0.0117351, 0.0122704, 0.0117171, 0.0123217, -0.0003209, 0.0002894
3: -0.0025434, -0.0019898, -0.0025621, -0.0019367, -0.0003319, 0.0002994
4: -0.0018829, -0.0012835, -0.0019403, -0.0012633, -0.0003241, 0.0003593
5: 0.0053280, 0.0058952, 0.0053089, 0.0059495, -0.0003400, 0.0003067
6: -0.0011603, 0.0010901, -0.0012363, 0.0013057, -0.0013492, 0.0012168
7: -0.0040413, -0.0009764, -0.0043350, -0.0008730, -0.0016572, 0.0018374
8: 0.9863671, 0.9885260, 0.9861602, 0.9885989, -0.0011674, 0.0012943
9: -0.0054720, -0.0035122, -0.0055381, -0.0033244, -0.0011749, 0.0010597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005327
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005275
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037930, 0.0047711, 0.0036952, 0.0047799, -0.0005248, 0.0006039
1: 0.0018703, 0.0020116, 0.0018562, 0.0020129, -0.0000758, 0.0000872
2: 0.0117220, 0.0122628, 0.0117172, 0.0123169, -0.0003339, 0.0002902
3: -0.0025570, -0.0019977, -0.0025620, -0.0019418, -0.0003453, 0.0003001
4: -0.0018743, -0.0012689, -0.0019349, -0.0012635, -0.0003249, 0.0003738
5: 0.0053141, 0.0058871, 0.0053090, 0.0059444, -0.0003537, 0.0003074
6: -0.0012154, 0.0010580, -0.0012357, 0.0012853, -0.0014036, 0.0012198
7: -0.0039976, -0.0009014, -0.0043072, -0.0008738, -0.0016613, 0.0019115
8: 0.9863979, 0.9885790, 0.9861797, 0.9885983, -0.0011703, 0.0013465
9: -0.0055200, -0.0035402, -0.0055376, -0.0033422, -0.0012223, 0.0010623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005651, upper bound: 0.0005327
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005654, upper bound: 0.0005275
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037792, 0.0047474, 0.0036808, 0.0047985, -0.0004962, 0.0005397
1: 0.0018683, 0.0020082, 0.0018541, 0.0020155, -0.0000717, 0.0000780
2: 0.0117351, 0.0122704, 0.0117069, 0.0123249, -0.0002984, 0.0002744
3: -0.0025434, -0.0019898, -0.0025726, -0.0019335, -0.0003086, 0.0002838
4: -0.0018829, -0.0012835, -0.0019438, -0.0012520, -0.0003072, 0.0003341
5: 0.0053280, 0.0058952, 0.0052981, 0.0059529, -0.0003162, 0.0002907
6: -0.0011603, 0.0010901, -0.0012790, 0.0013189, -0.0012545, 0.0011534
7: -0.0040413, -0.0009764, -0.0043529, -0.0008149, -0.0015708, 0.0017085
8: 0.9863671, 0.9885260, 0.9861476, 0.9886399, -0.0011065, 0.0012035
9: -0.0054720, -0.0035122, -0.0055753, -0.0033130, -0.0010925, 0.0010044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005327
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005275
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037930, 0.0047711, 0.0036904, 0.0047981, -0.0004961, 0.0005669
1: 0.0018703, 0.0020116, 0.0018555, 0.0020155, -0.0000717, 0.0000819
2: 0.0117220, 0.0122628, 0.0117071, 0.0123195, -0.0003134, 0.0002743
3: -0.0025570, -0.0019977, -0.0025724, -0.0019390, -0.0003242, 0.0002837
4: -0.0018743, -0.0012689, -0.0019378, -0.0012522, -0.0003071, 0.0003509
5: 0.0053141, 0.0058871, 0.0052983, 0.0059472, -0.0003321, 0.0002906
6: -0.0012154, 0.0010580, -0.0012781, 0.0012964, -0.0013177, 0.0011530
7: -0.0039976, -0.0009014, -0.0043223, -0.0008160, -0.0015703, 0.0017946
8: 0.9863979, 0.9885790, 0.9861692, 0.9886391, -0.0011062, 0.0012641
9: -0.0055200, -0.0035402, -0.0055746, -0.0033326, -0.0011475, 0.0010041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005651, upper bound: 0.0005327
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0005654, upper bound: 0.0005275
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0036906, 0.0047980, 0.0036864, 0.0047801, -0.0004689, 0.0004917
1: 0.0018555, 0.0020155, 0.0018549, 0.0020129, -0.0000677, 0.0000710
2: 0.0117072, 0.0123194, 0.0117171, 0.0123217, -0.0002718, 0.0002592
3: -0.0025723, -0.0019392, -0.0025621, -0.0019367, -0.0002811, 0.0002681
4: -0.0019377, -0.0012523, -0.0019403, -0.0012633, -0.0002903, 0.0003044
5: 0.0052984, 0.0059471, 0.0053089, 0.0059495, -0.0002880, 0.0002747
6: -0.0012778, 0.0012959, -0.0012363, 0.0013057, -0.0011428, 0.0010899
7: -0.0043217, -0.0008164, -0.0043350, -0.0008730, -0.0014843, 0.0015564
8: 0.9861696, 0.9886388, 0.9861602, 0.9885989, -0.0010456, 0.0010963
9: -0.0055743, -0.0033330, -0.0055381, -0.0033244, -0.0009952, 0.0009491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004685
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0037069, 0.0048189, 0.0036952, 0.0047799, -0.0004705, 0.0005176
1: 0.0018578, 0.0020185, 0.0018562, 0.0020129, -0.0000680, 0.0000748
2: 0.0116956, 0.0123104, 0.0117172, 0.0123169, -0.0002861, 0.0002601
3: -0.0025843, -0.0019484, -0.0025620, -0.0019418, -0.0002959, 0.0002690
4: -0.0019277, -0.0012393, -0.0019349, -0.0012635, -0.0002912, 0.0003204
5: 0.0052861, 0.0059376, 0.0053090, 0.0059444, -0.0003032, 0.0002756
6: -0.0013266, 0.0012582, -0.0012357, 0.0012853, -0.0012030, 0.0010935
7: -0.0042703, -0.0007500, -0.0043072, -0.0008738, -0.0014892, 0.0016383
8: 0.9862058, 0.9886855, 0.9861797, 0.9885983, -0.0010490, 0.0011541
9: -0.0056167, -0.0033658, -0.0055376, -0.0033422, -0.0010476, 0.0009522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004685
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036906, 0.0047980, 0.0036808, 0.0047985, -0.0004439, 0.0004538
1: 0.0018555, 0.0020155, 0.0018541, 0.0020155, -0.0000641, 0.0000656
2: 0.0117072, 0.0123194, 0.0117069, 0.0123249, -0.0002509, 0.0002454
3: -0.0025723, -0.0019392, -0.0025726, -0.0019335, -0.0002595, 0.0002538
4: -0.0019377, -0.0012523, -0.0019438, -0.0012520, -0.0002748, 0.0002809
5: 0.0052984, 0.0059471, 0.0052981, 0.0059529, -0.0002659, 0.0002600
6: -0.0012778, 0.0012959, -0.0012790, 0.0013189, -0.0010549, 0.0010317
7: -0.0043217, -0.0008164, -0.0043529, -0.0008149, -0.0014050, 0.0014366
8: 0.9861696, 0.9886388, 0.9861476, 0.9886399, -0.0009897, 0.0010120
9: -0.0055743, -0.0033330, -0.0055753, -0.0033130, -0.0009186, 0.0008984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006508, upper bound: 0.0004683
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004654
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0037069, 0.0048189, 0.0036904, 0.0047981, -0.0004440, 0.0004836
1: 0.0018578, 0.0020185, 0.0018555, 0.0020155, -0.0000641, 0.0000699
2: 0.0116956, 0.0123104, 0.0117071, 0.0123195, -0.0002674, 0.0002455
3: -0.0025843, -0.0019484, -0.0025724, -0.0019390, -0.0002765, 0.0002539
4: -0.0019277, -0.0012393, -0.0019378, -0.0012522, -0.0002748, 0.0002994
5: 0.0052861, 0.0059376, 0.0052983, 0.0059472, -0.0002833, 0.0002601
6: -0.0013266, 0.0012582, -0.0012781, 0.0012964, -0.0011240, 0.0010320
7: -0.0042703, -0.0007500, -0.0043223, -0.0008160, -0.0014055, 0.0015308
8: 0.9862058, 0.9886855, 0.9861692, 0.9886391, -0.0009901, 0.0010783
9: -0.0056167, -0.0033658, -0.0055746, -0.0033326, -0.0009788, 0.0008987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004683
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004654
time: 0.63 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.06 seconds
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004766, upper bound: 0.0005926
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004769, upper bound: 0.0005808
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004784, upper bound: 0.0005926
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004786, upper bound: 0.0005808
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004758, upper bound: 0.0005964
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004760, upper bound: 0.0005813
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004776, upper bound: 0.0005964
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004776, upper bound: 0.0005813
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0005789
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005254, upper bound: 0.0005703
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005276, upper bound: 0.0005789
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005703
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005250, upper bound: 0.0005753
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005250, upper bound: 0.0005649
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005272, upper bound: 0.0005755
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005275, upper bound: 0.0005654
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005924
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004883, upper bound: 0.0005789
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005924
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005789
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005906
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004883, upper bound: 0.0005758
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005906
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005758
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005787
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005399, upper bound: 0.0005691
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005787
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005424, upper bound: 0.0005691
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005747
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005399, upper bound: 0.0005641
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005747
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005424, upper bound: 0.0005641
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004822, upper bound: 0.0005686
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004833, upper bound: 0.0005658
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004871, upper bound: 0.0005686
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004877, upper bound: 0.0005658
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004631, upper bound: 0.0006453
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004636, upper bound: 0.0006447
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004680, upper bound: 0.0006453
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004680, upper bound: 0.0006447
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004629, upper bound: 0.0006534
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004631, upper bound: 0.0006502
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004676, upper bound: 0.0006535
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004676, upper bound: 0.0006502
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005255, upper bound: 0.0005798
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005255, upper bound: 0.0005727
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005798
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005280, upper bound: 0.0005727
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005252, upper bound: 0.0005762
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005252, upper bound: 0.0005676
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005275, upper bound: 0.0005765
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005684
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004711, upper bound: 0.0006451
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004715, upper bound: 0.0006441
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006451
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006441
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004711, upper bound: 0.0006441
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004715, upper bound: 0.0006421
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006441
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006421
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005797
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005402, upper bound: 0.0005721
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005797
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005427, upper bound: 0.0005721
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005757
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005402, upper bound: 0.0005669
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005757
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005427, upper bound: 0.0005669
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005814, upper bound: 0.0004834
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005816, upper bound: 0.0004785
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005806, upper bound: 0.0004834
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005808, upper bound: 0.0004785
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005800, upper bound: 0.0004990
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0004902
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0004990
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0004902
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004712
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004680
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004712
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004680
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004768
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004718
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004768
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004717
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005814, upper bound: 0.0004834
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005816, upper bound: 0.0004785
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005806, upper bound: 0.0004834
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005808, upper bound: 0.0004786
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005800, upper bound: 0.0004990
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005801, upper bound: 0.0004902
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0004990
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0004902
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004712
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004680
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004711
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004680
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004768
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004718
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004768
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004717
IS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004832
IS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004776
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004832
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004776
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004828
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004773
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004828
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004773
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004677
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004677
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004645
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004645
IS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004832
IS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004776
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004832
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004776
IS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004828
IS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005841, upper bound: 0.0004773
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004828
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0004773
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004677
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004677
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004645
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004645
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005697, upper bound: 0.0005328
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005711, upper bound: 0.0005277
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005684, upper bound: 0.0005328
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005703, upper bound: 0.0005277
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005689, upper bound: 0.0005495
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005699, upper bound: 0.0005424
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005675, upper bound: 0.0005495
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005691, upper bound: 0.0005424
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004715
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004686
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004715
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004686
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004768
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004725
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004768
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004725
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005697, upper bound: 0.0005328
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005711, upper bound: 0.0005277
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005684, upper bound: 0.0005328
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005703, upper bound: 0.0005277
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005689, upper bound: 0.0005495
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005699, upper bound: 0.0005424
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005675, upper bound: 0.0005495
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005691, upper bound: 0.0005424
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004715
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004686
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004715
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004686
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004768
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004725
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004768
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004725
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005327
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005275
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005651, upper bound: 0.0005327
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005654, upper bound: 0.0005275
IS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005327
IS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005275
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005651, upper bound: 0.0005327
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005654, upper bound: 0.0005275
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004685
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004685
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004682
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004654
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004683
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004654
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005327
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005275
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005651, upper bound: 0.0005327
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005654, upper bound: 0.0005275
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005327
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005275
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005651, upper bound: 0.0005327
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0005654, upper bound: 0.0005275
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004685
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004685
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006508, upper bound: 0.0004683
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004654
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004683
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004654

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038944, 0.0046407, 0.0037803, 0.0047337, -0.0004757, 0.0005022
1: 0.0018849, 0.0019928, 0.0018684, 0.0020062, -0.0000687, 0.0000725
2: 0.0117941, 0.0122068, 0.0117427, 0.0122698, -0.0002776, 0.0002630
3: -0.0024824, -0.0020557, -0.0025356, -0.0019904, -0.0002871, 0.0002720
4: -0.0018116, -0.0013496, -0.0018822, -0.0012921, -0.0002945, 0.0003108
5: 0.0053905, 0.0058277, 0.0053361, 0.0058946, -0.0002942, 0.0002787
6: -0.0009124, 0.0008224, -0.0011284, 0.0010875, -0.0011672, 0.0011056
7: -0.0036767, -0.0013142, -0.0040378, -0.0010199, -0.0015058, 0.0015896
8: 0.9866240, 0.9882881, 0.9863696, 0.9884953, -0.0010607, 0.0011197
9: -0.0052560, -0.0037454, -0.0054442, -0.0035144, -0.0010164, 0.0009628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004766, upper bound: 0.0005806
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004766, upper bound: 0.0005808
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039055, 0.0046692, 0.0037894, 0.0047333, -0.0004766, 0.0005302
1: 0.0018865, 0.0019969, 0.0018698, 0.0020061, -0.0000689, 0.0000766
2: 0.0117784, 0.0122006, 0.0117429, 0.0122648, -0.0002932, 0.0002635
3: -0.0024987, -0.0020620, -0.0025354, -0.0019956, -0.0003032, 0.0002725
4: -0.0018047, -0.0013320, -0.0018766, -0.0012923, -0.0002950, 0.0003282
5: 0.0053738, 0.0058212, 0.0053363, 0.0058892, -0.0003106, 0.0002792
6: -0.0009786, 0.0007965, -0.0011276, 0.0010665, -0.0012324, 0.0011078
7: -0.0036415, -0.0012240, -0.0040092, -0.0010210, -0.0015087, 0.0016785
8: 0.9866487, 0.9883516, 0.9863898, 0.9884946, -0.0010628, 0.0011823
9: -0.0053137, -0.0037679, -0.0054435, -0.0035328, -0.0010733, 0.0009647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004784, upper bound: 0.0005806
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004784, upper bound: 0.0005808
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038944, 0.0046407, 0.0037771, 0.0047479, -0.0004918, 0.0005052
1: 0.0018849, 0.0019928, 0.0018680, 0.0020082, -0.0000710, 0.0000730
2: 0.0117941, 0.0122068, 0.0117349, 0.0122716, -0.0002793, 0.0002719
3: -0.0024824, -0.0020557, -0.0025437, -0.0019886, -0.0002889, 0.0002812
4: -0.0018116, -0.0013496, -0.0018842, -0.0012833, -0.0003044, 0.0003127
5: 0.0053905, 0.0058277, 0.0053278, 0.0058964, -0.0002959, 0.0002881
6: -0.0009124, 0.0008224, -0.0011614, 0.0010950, -0.0011742, 0.0011430
7: -0.0036767, -0.0013142, -0.0040480, -0.0009751, -0.0015567, 0.0015991
8: 0.9866240, 0.9882881, 0.9863623, 0.9885271, -0.0010966, 0.0011264
9: -0.0052560, -0.0037454, -0.0054729, -0.0035079, -0.0010225, 0.0009954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004758, upper bound: 0.0005813
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004758, upper bound: 0.0005813
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039055, 0.0046692, 0.0037860, 0.0047475, -0.0004927, 0.0005331
1: 0.0018865, 0.0019969, 0.0018693, 0.0020082, -0.0000712, 0.0000770
2: 0.0117784, 0.0122006, 0.0117351, 0.0122667, -0.0002947, 0.0002724
3: -0.0024987, -0.0020620, -0.0025435, -0.0019937, -0.0003048, 0.0002817
4: -0.0018047, -0.0013320, -0.0018787, -0.0012835, -0.0003050, 0.0003300
5: 0.0053738, 0.0058212, 0.0053280, 0.0058912, -0.0003123, 0.0002886
6: -0.0009786, 0.0007965, -0.0011604, 0.0010744, -0.0012390, 0.0011451
7: -0.0036415, -0.0012240, -0.0040199, -0.0009763, -0.0015595, 0.0016874
8: 0.9866487, 0.9883516, 0.9863821, 0.9885261, -0.0010986, 0.0011886
9: -0.0053137, -0.0037679, -0.0054720, -0.0035259, -0.0010789, 0.0009972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004776, upper bound: 0.0005813
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004776, upper bound: 0.0005813
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038891, 0.0046562, 0.0037803, 0.0047337, -0.0005130, 0.0005505
1: 0.0018842, 0.0019950, 0.0018684, 0.0020062, -0.0000741, 0.0000795
2: 0.0117856, 0.0122097, 0.0117427, 0.0122698, -0.0003044, 0.0002836
3: -0.0024913, -0.0020526, -0.0025356, -0.0019904, -0.0003148, 0.0002933
4: -0.0018149, -0.0013400, -0.0018822, -0.0012921, -0.0003176, 0.0003408
5: 0.0053815, 0.0058308, 0.0053361, 0.0058946, -0.0003225, 0.0003005
6: -0.0009482, 0.0008348, -0.0011284, 0.0010875, -0.0012795, 0.0011924
7: -0.0036936, -0.0012653, -0.0040378, -0.0010199, -0.0016239, 0.0017426
8: 0.9866120, 0.9883226, 0.9863696, 0.9884953, -0.0011439, 0.0012275
9: -0.0052873, -0.0037346, -0.0054442, -0.0035144, -0.0011142, 0.0010384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005789
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005789
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039002, 0.0046848, 0.0037894, 0.0047333, -0.0005139, 0.0005742
1: 0.0018858, 0.0019991, 0.0018698, 0.0020061, -0.0000742, 0.0000830
2: 0.0117698, 0.0122036, 0.0117429, 0.0122648, -0.0003175, 0.0002841
3: -0.0025076, -0.0020590, -0.0025354, -0.0019956, -0.0003283, 0.0002939
4: -0.0018080, -0.0013223, -0.0018766, -0.0012923, -0.0003181, 0.0003555
5: 0.0053647, 0.0058243, 0.0053363, 0.0058892, -0.0003364, 0.0003011
6: -0.0010147, 0.0008089, -0.0011276, 0.0010665, -0.0013346, 0.0011945
7: -0.0036584, -0.0011748, -0.0040092, -0.0010210, -0.0016269, 0.0018177
8: 0.9866368, 0.9883863, 0.9863898, 0.9884946, -0.0011460, 0.0012804
9: -0.0053452, -0.0037571, -0.0054435, -0.0035328, -0.0011623, 0.0010403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005789
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005789
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0038891, 0.0046562, 0.0037771, 0.0047479, -0.0004848, 0.0005111
1: 0.0018842, 0.0019950, 0.0018680, 0.0020082, -0.0000700, 0.0000738
2: 0.0117856, 0.0122097, 0.0117349, 0.0122716, -0.0002826, 0.0002680
3: -0.0024913, -0.0020526, -0.0025437, -0.0019886, -0.0002923, 0.0002772
4: -0.0018149, -0.0013400, -0.0018842, -0.0012833, -0.0003001, 0.0003164
5: 0.0053815, 0.0058308, 0.0053278, 0.0058964, -0.0002994, 0.0002840
6: -0.0009482, 0.0008348, -0.0011614, 0.0010950, -0.0011880, 0.0011268
7: -0.0036936, -0.0012653, -0.0040480, -0.0009751, -0.0015346, 0.0016180
8: 0.9866120, 0.9883226, 0.9863623, 0.9885271, -0.0010810, 0.0011397
9: -0.0052873, -0.0037346, -0.0054729, -0.0035079, -0.0010346, 0.0009812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005757
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005758
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039002, 0.0046848, 0.0037860, 0.0047475, -0.0004856, 0.0005397
1: 0.0018858, 0.0019991, 0.0018693, 0.0020082, -0.0000702, 0.0000780
2: 0.0117698, 0.0122036, 0.0117351, 0.0122667, -0.0002984, 0.0002685
3: -0.0025076, -0.0020590, -0.0025435, -0.0019937, -0.0003086, 0.0002777
4: -0.0018080, -0.0013223, -0.0018787, -0.0012835, -0.0003006, 0.0003341
5: 0.0053647, 0.0058243, 0.0053280, 0.0058912, -0.0003162, 0.0002845
6: -0.0010147, 0.0008089, -0.0011604, 0.0010744, -0.0012544, 0.0011287
7: -0.0036584, -0.0011748, -0.0040199, -0.0009763, -0.0015372, 0.0017084
8: 0.9866368, 0.9883863, 0.9863821, 0.9885261, -0.0010829, 0.0012035
9: -0.0053452, -0.0037571, -0.0054720, -0.0035259, -0.0010924, 0.0009830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005757
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005758
time: 0.54 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.93 seconds
IS_A1_B1_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004766, upper bound: 0.0005806
IS_A1_B1_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004766, upper bound: 0.0005808
IS_A1_B1_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004784, upper bound: 0.0005806
IS_A1_B1_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004784, upper bound: 0.0005808
IS_A1_B1_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004758, upper bound: 0.0005813
IS_A1_B1_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004758, upper bound: 0.0005813
IS_A1_B1_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004776, upper bound: 0.0005813
IS_A1_B1_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004776, upper bound: 0.0005813
IS_A1_B1_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005789
IS_A1_B1_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005789
IS_A1_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005789
IS_A1_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005789
IS_A1_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005757
IS_A1_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004879, upper bound: 0.0005758
IS_A1_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005757
IS_A1_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.93
Output dim: 8, lower bound: -0.0004902, upper bound: 0.0005758
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004631, upper bound: 0.0006453
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004636, upper bound: 0.0006447
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004680, upper bound: 0.0006453
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004680, upper bound: 0.0006447
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004629, upper bound: 0.0006534
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004631, upper bound: 0.0006502
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004676, upper bound: 0.0006535
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004676, upper bound: 0.0006502
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004711, upper bound: 0.0006451
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004715, upper bound: 0.0006441
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006451
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006441
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004711, upper bound: 0.0006441
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004715, upper bound: 0.0006421
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006441
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0004717, upper bound: 0.0006421
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004712
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004680
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004712
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004680
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004768
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004718
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004768
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004717
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004712
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006415, upper bound: 0.0004680
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004711
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006447, upper bound: 0.0004680
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004768
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006410, upper bound: 0.0004718
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004768
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006441, upper bound: 0.0004717
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004677
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004677
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004645
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004645
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004677
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004711
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004677
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006504, upper bound: 0.0004645
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004681
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006502, upper bound: 0.0004645
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004715
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004686
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004715
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004686
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004768
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004725
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004768
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004725
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004715
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006420, upper bound: 0.0004686
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004715
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006452, upper bound: 0.0004686
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004768
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006414, upper bound: 0.0004725
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004768
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006446, upper bound: 0.0004725
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004685
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004685
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004682
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004654
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004683
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004654
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004685
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004715
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004685
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006508, upper bound: 0.0004683
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006509, upper bound: 0.0004654
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004683
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 8, lower bound: -0.0006507, upper bound: 0.0004654

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.92 + 597.97 = 600.89 seconds
