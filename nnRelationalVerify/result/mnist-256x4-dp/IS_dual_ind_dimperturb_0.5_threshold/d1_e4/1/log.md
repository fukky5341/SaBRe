## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0012732


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0023399, 0.0050233, 0.0023399, 0.0050233, -0.0016962, 0.0016962)
1: (0.0016604, 0.0020480, 0.0016604, 0.0020480, -0.0002450, 0.0002450)
2: (0.0115826, 0.0130662, 0.0115826, 0.0130662, -0.0009378, 0.0009378)
3: (-0.0027012, -0.0011668, -0.0027012, -0.0011668, -0.0009699, 0.0009699)
4: (-0.0027738, -0.0011128, -0.0027738, -0.0011128, -0.0010500, 0.0010500)
5: (0.0051664, 0.0067383, 0.0051664, 0.0067383, -0.0009936, 0.0009936)
6: (-0.0018015, 0.0044354, -0.0018015, 0.0044354, -0.0039424, 0.0039424)
7: (-0.0085973, -0.0001032, -0.0085973, -0.0001032, -0.0053692, 0.0053692)
8: (0.9831578, 0.9891411, 0.9831578, 0.9891411, -0.0037822, 0.0037822)
9: (-0.0060304, -0.0005990, -0.0060304, -0.0005990, -0.0034332, 0.0034332)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 1.68 = 3.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0021964, upper bound: 0.0021964

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020854, upper bound: 0.0020802
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020802, upper bound: 0.0020802
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 8, lower bound: -0.0020854, upper bound: 0.0020802
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 8, lower bound: -0.0020802, upper bound: 0.0020802

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0024422, 0.0050166, 0.0023805, 0.0050208, -0.0015497, 0.0016308
1: 0.0016751, 0.0020471, 0.0016662, 0.0020477, -0.0002239, 0.0002356
2: 0.0115863, 0.0130096, 0.0115840, 0.0130438, -0.0009016, 0.0008568
3: -0.0026973, -0.0012253, -0.0026997, -0.0011900, -0.0009325, 0.0008861
4: -0.0027105, -0.0011169, -0.0027487, -0.0011143, -0.0009593, 0.0010095
5: 0.0051703, 0.0066784, 0.0051679, 0.0067146, -0.0009553, 0.0009078
6: -0.0017859, 0.0041976, -0.0017956, 0.0043412, -0.0037904, 0.0036020
7: -0.0082735, -0.0001245, -0.0084690, -0.0001112, -0.0049056, 0.0051621
8: 0.9833859, 0.9891262, 0.9832481, 0.9891354, -0.0034556, 0.0036363
9: -0.0060168, -0.0008061, -0.0060252, -0.0006810, -0.0033008, 0.0031368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020026, upper bound: 0.0019830
time: 0.72 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020050, upper bound: 0.0019995
time: 0.80 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0024964, 0.0050916, 0.0024260, 0.0050196, -0.0015516, 0.0018656
1: 0.0016830, 0.0020579, 0.0016728, 0.0020475, -0.0002242, 0.0002695
2: 0.0115448, 0.0129797, 0.0115847, 0.0130186, -0.0010314, 0.0008579
3: -0.0027402, -0.0012563, -0.0026991, -0.0012160, -0.0010668, 0.0008872
4: -0.0026770, -0.0010705, -0.0027205, -0.0011151, -0.0009605, 0.0011548
5: 0.0051264, 0.0066467, 0.0051686, 0.0066879, -0.0010929, 0.0009089
6: -0.0019603, 0.0040717, -0.0017929, 0.0042352, -0.0043362, 0.0036064
7: -0.0081021, 0.0001130, -0.0083247, -0.0001150, -0.0049116, 0.0059055
8: 0.9835065, 0.9892935, 0.9833497, 0.9891329, -0.0034598, 0.0041600
9: -0.0061686, -0.0009157, -0.0060228, -0.0007733, -0.0037762, 0.0031406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019970, upper bound: 0.0019830
time: 0.92 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019995, upper bound: 0.0019995
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.09 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 8, lower bound: -0.0020026, upper bound: 0.0019830
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 8, lower bound: -0.0020050, upper bound: 0.0019995
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 8, lower bound: -0.0019970, upper bound: 0.0019830
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 8, lower bound: -0.0019995, upper bound: 0.0019995

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0024993, 0.0050147, 0.0025092, 0.0050167, -0.0014730, 0.0014734
1: 0.0016834, 0.0020468, 0.0016848, 0.0020471, -0.0002128, 0.0002129
2: 0.0115873, 0.0129780, 0.0115862, 0.0129726, -0.0008146, 0.0008144
3: -0.0026963, -0.0012579, -0.0026974, -0.0012636, -0.0008425, 0.0008423
4: -0.0026752, -0.0011181, -0.0026690, -0.0011168, -0.0009118, 0.0009120
5: 0.0051714, 0.0066449, 0.0051702, 0.0066392, -0.0008631, 0.0008629
6: -0.0017816, 0.0040649, -0.0017863, 0.0040419, -0.0034245, 0.0034236
7: -0.0080927, -0.0001303, -0.0080615, -0.0001239, -0.0046626, 0.0046639
8: 0.9835132, 0.9891222, 0.9835352, 0.9891265, -0.0032845, 0.0032854
9: -0.0060131, -0.0009216, -0.0060171, -0.0009416, -0.0029822, 0.0029814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019216
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019064
time: 0.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0024849, 0.0050152, 0.0024703, 0.0050381, -0.0016284, 0.0014885
1: 0.0016813, 0.0020468, 0.0016792, 0.0020502, -0.0002353, 0.0002150
2: 0.0115871, 0.0129861, 0.0115744, 0.0129941, -0.0008229, 0.0009003
3: -0.0026965, -0.0012497, -0.0027097, -0.0012414, -0.0008511, 0.0009312
4: -0.0026841, -0.0011178, -0.0026931, -0.0011036, -0.0010080, 0.0009214
5: 0.0051712, 0.0066534, 0.0051577, 0.0066619, -0.0008719, 0.0009539
6: -0.0017827, 0.0040985, -0.0018360, 0.0041323, -0.0034596, 0.0037849
7: -0.0081385, -0.0001289, -0.0081845, -0.0000562, -0.0051548, 0.0047116
8: 0.9834809, 0.9891231, 0.9834486, 0.9891742, -0.0036311, 0.0033190
9: -0.0060139, -0.0008924, -0.0060604, -0.0008630, -0.0030128, 0.0032961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019392
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019213
time: 0.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025546, 0.0050896, 0.0025584, 0.0050154, -0.0014752, 0.0017095
1: 0.0016914, 0.0020576, 0.0016919, 0.0020469, -0.0002131, 0.0002470
2: 0.0115460, 0.0129475, 0.0115870, 0.0129454, -0.0009451, 0.0008156
3: -0.0027391, -0.0012895, -0.0026967, -0.0012917, -0.0009775, 0.0008436
4: -0.0026410, -0.0010717, -0.0026386, -0.0011176, -0.0009132, 0.0010582
5: 0.0051276, 0.0066126, 0.0051710, 0.0066103, -0.0010014, 0.0008642
6: -0.0019556, 0.0039365, -0.0017833, 0.0039275, -0.0039734, 0.0034289
7: -0.0079178, 0.0001067, -0.0079056, -0.0001281, -0.0046698, 0.0054114
8: 0.9836363, 0.9892890, 0.9836450, 0.9891236, -0.0032895, 0.0038119
9: -0.0061646, -0.0010335, -0.0060145, -0.0010413, -0.0034602, 0.0029860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019191, upper bound: 0.0019216
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019191, upper bound: 0.0019063
time: 0.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025384, 0.0050902, 0.0025155, 0.0050370, -0.0016296, 0.0017247
1: 0.0016890, 0.0020577, 0.0016857, 0.0020500, -0.0002354, 0.0002492
2: 0.0115456, 0.0129564, 0.0115750, 0.0129691, -0.0009535, 0.0009009
3: -0.0027395, -0.0012803, -0.0027090, -0.0012672, -0.0009862, 0.0009318
4: -0.0026510, -0.0010713, -0.0026651, -0.0011043, -0.0010087, 0.0010676
5: 0.0051272, 0.0066221, 0.0051584, 0.0066355, -0.0010103, 0.0009546
6: -0.0019571, 0.0039741, -0.0018334, 0.0040272, -0.0040087, 0.0037875
7: -0.0079690, 0.0001087, -0.0080414, -0.0000598, -0.0051583, 0.0054594
8: 0.9836003, 0.9892905, 0.9835493, 0.9891718, -0.0036336, 0.0038457
9: -0.0061659, -0.0010007, -0.0060581, -0.0009544, -0.0034909, 0.0032984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019213, upper bound: 0.0019392
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019213, upper bound: 0.0019213
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.98 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019216
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019064
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019392
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019213
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.0019191, upper bound: 0.0019216
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.0019191, upper bound: 0.0019063
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.0019213, upper bound: 0.0019392
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.0019213, upper bound: 0.0019213

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0025080, 0.0049535, 0.0025125, 0.0049932, -0.0014379, 0.0014047
1: 0.0016846, 0.0020379, 0.0016853, 0.0020437, -0.0002077, 0.0002029
2: 0.0116212, 0.0129733, 0.0115993, 0.0129708, -0.0007766, 0.0007950
3: -0.0026613, -0.0012629, -0.0026839, -0.0012655, -0.0008032, 0.0008222
4: -0.0026698, -0.0011560, -0.0026670, -0.0011314, -0.0008901, 0.0008695
5: 0.0052073, 0.0066399, 0.0051841, 0.0066372, -0.0008229, 0.0008423
6: -0.0016393, 0.0040447, -0.0017315, 0.0040342, -0.0032649, 0.0033421
7: -0.0080653, -0.0003241, -0.0080510, -0.0001986, -0.0045517, 0.0044465
8: 0.9835325, 0.9889856, 0.9835426, 0.9890740, -0.0032063, 0.0031322
9: -0.0058891, -0.0009392, -0.0059694, -0.0009483, -0.0028432, 0.0029105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019131
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019216
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0024860, 0.0049146, 0.0025137, 0.0049636, -0.0014900, 0.0013929
1: 0.0016815, 0.0020323, 0.0016855, 0.0020394, -0.0002153, 0.0002012
2: 0.0116427, 0.0129854, 0.0116156, 0.0129701, -0.0007701, 0.0008238
3: -0.0026390, -0.0012503, -0.0026670, -0.0012662, -0.0007964, 0.0008520
4: -0.0026834, -0.0011801, -0.0026662, -0.0011498, -0.0009223, 0.0008622
5: 0.0052301, 0.0066528, 0.0052014, 0.0066365, -0.0008159, 0.0008728
6: -0.0015488, 0.0040959, -0.0016627, 0.0040314, -0.0032374, 0.0034631
7: -0.0081349, -0.0004474, -0.0080472, -0.0002923, -0.0047164, 0.0044090
8: 0.9834835, 0.9888988, 0.9835452, 0.9890080, -0.0033224, 0.0031058
9: -0.0058103, -0.0008947, -0.0059095, -0.0009508, -0.0028193, 0.0030158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0018972
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019064
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0024936, 0.0049540, 0.0024737, 0.0050149, -0.0015918, 0.0014197
1: 0.0016825, 0.0020380, 0.0016797, 0.0020468, -0.0002300, 0.0002051
2: 0.0116210, 0.0129812, 0.0115872, 0.0129922, -0.0007849, 0.0008801
3: -0.0026615, -0.0012547, -0.0026964, -0.0012433, -0.0008118, 0.0009102
4: -0.0026787, -0.0011557, -0.0026910, -0.0011179, -0.0009853, 0.0008788
5: 0.0052070, 0.0066483, 0.0051713, 0.0066600, -0.0008317, 0.0009325
6: -0.0016403, 0.0040783, -0.0017821, 0.0041244, -0.0032998, 0.0036998
7: -0.0081109, -0.0003227, -0.0081738, -0.0001296, -0.0050388, 0.0044941
8: 0.9835003, 0.9889866, 0.9834561, 0.9891226, -0.0035494, 0.0031657
9: -0.0058900, -0.0009100, -0.0060135, -0.0008698, -0.0028736, 0.0032219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019325
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019392
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0024749, 0.0049151, 0.0024750, 0.0049837, -0.0016491, 0.0014083
1: 0.0016799, 0.0020324, 0.0016799, 0.0020423, -0.0002382, 0.0002035
2: 0.0116425, 0.0129916, 0.0116045, 0.0129915, -0.0007786, 0.0009117
3: -0.0026393, -0.0012440, -0.0026785, -0.0012440, -0.0008053, 0.0009430
4: -0.0026903, -0.0011798, -0.0026902, -0.0011373, -0.0010208, 0.0008718
5: 0.0052298, 0.0066593, 0.0051896, 0.0066592, -0.0008250, 0.0009660
6: -0.0015500, 0.0041217, -0.0017094, 0.0041214, -0.0032734, 0.0038329
7: -0.0081701, -0.0004458, -0.0081697, -0.0002286, -0.0052201, 0.0044581
8: 0.9834586, 0.9888998, 0.9834589, 0.9890529, -0.0036771, 0.0031404
9: -0.0058113, -0.0008722, -0.0059502, -0.0008724, -0.0028506, 0.0033379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019140
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019213
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0025631, 0.0050292, 0.0025618, 0.0049919, -0.0014404, 0.0016450
1: 0.0016926, 0.0020489, 0.0016924, 0.0020435, -0.0002081, 0.0002376
2: 0.0115793, 0.0129428, 0.0116000, 0.0129435, -0.0009095, 0.0007964
3: -0.0027046, -0.0012944, -0.0026832, -0.0012936, -0.0009406, 0.0008236
4: -0.0026357, -0.0011091, -0.0026365, -0.0011322, -0.0008916, 0.0010183
5: 0.0051629, 0.0066076, 0.0051848, 0.0066084, -0.0009636, 0.0008438
6: -0.0018153, 0.0039166, -0.0017285, 0.0039198, -0.0038233, 0.0033479
7: -0.0078907, -0.0000845, -0.0078951, -0.0002027, -0.0045596, 0.0052071
8: 0.9836555, 0.9891545, 0.9836524, 0.9890711, -0.0032118, 0.0036680
9: -0.0060423, -0.0010508, -0.0059667, -0.0010480, -0.0033295, 0.0029155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018362, upper bound: 0.0017742
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018539, upper bound: 0.0018589
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025415, 0.0049910, 0.0025630, 0.0049622, -0.0014687, 0.0016009
1: 0.0016895, 0.0020433, 0.0016926, 0.0020392, -0.0002122, 0.0002313
2: 0.0116005, 0.0129548, 0.0116164, 0.0129429, -0.0008851, 0.0008120
3: -0.0026827, -0.0012820, -0.0026662, -0.0012944, -0.0009154, 0.0008398
4: -0.0026491, -0.0011328, -0.0026357, -0.0011506, -0.0009092, 0.0009910
5: 0.0051854, 0.0066203, 0.0052022, 0.0066077, -0.0009378, 0.0008604
6: -0.0017264, 0.0039669, -0.0016595, 0.0039169, -0.0037210, 0.0034138
7: -0.0079593, -0.0002056, -0.0078912, -0.0002966, -0.0046493, 0.0050677
8: 0.9836072, 0.9890691, 0.9836552, 0.9890049, -0.0032750, 0.0035698
9: -0.0059649, -0.0010069, -0.0059067, -0.0010505, -0.0032404, 0.0029729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018362, upper bound: 0.0017618
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018539, upper bound: 0.0018444
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0025471, 0.0050298, 0.0025189, 0.0050138, -0.0015612, 0.0016601
1: 0.0016903, 0.0020490, 0.0016862, 0.0020467, -0.0002255, 0.0002398
2: 0.0115790, 0.0129516, 0.0115879, 0.0129672, -0.0009178, 0.0008631
3: -0.0027049, -0.0012853, -0.0026958, -0.0012692, -0.0009493, 0.0008927
4: -0.0026456, -0.0011087, -0.0026630, -0.0011186, -0.0009664, 0.0010276
5: 0.0051626, 0.0066170, 0.0051720, 0.0066335, -0.0009725, 0.0009145
6: -0.0018167, 0.0039538, -0.0017795, 0.0040193, -0.0038585, 0.0036286
7: -0.0079415, -0.0000825, -0.0080307, -0.0001332, -0.0049418, 0.0052549
8: 0.9836197, 0.9891557, 0.9835569, 0.9891201, -0.0034811, 0.0037017
9: -0.0060436, -0.0010183, -0.0060112, -0.0009613, -0.0033601, 0.0031599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018376, upper bound: 0.0017929
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018563, upper bound: 0.0018741
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025282, 0.0049915, 0.0025202, 0.0049825, -0.0016274, 0.0016162
1: 0.0016876, 0.0020434, 0.0016864, 0.0020421, -0.0002351, 0.0002335
2: 0.0116002, 0.0129621, 0.0116052, 0.0129665, -0.0008936, 0.0008998
3: -0.0026830, -0.0012745, -0.0026779, -0.0012699, -0.0009242, 0.0009306
4: -0.0026573, -0.0011324, -0.0026622, -0.0011380, -0.0010074, 0.0010005
5: 0.0051850, 0.0066280, 0.0051903, 0.0066327, -0.0009468, 0.0009533
6: -0.0017277, 0.0039977, -0.0017067, 0.0040164, -0.0037566, 0.0037826
7: -0.0080013, -0.0002037, -0.0080267, -0.0002323, -0.0051516, 0.0051161
8: 0.9835776, 0.9890704, 0.9835597, 0.9890503, -0.0036289, 0.0036039
9: -0.0059661, -0.0009801, -0.0059478, -0.0009639, -0.0032714, 0.0032941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018376, upper bound: 0.0017807
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018563, upper bound: 0.0018563
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.97 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019131
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019216
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0018972
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0019244, upper bound: 0.0019064
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019325
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019392
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019140
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0019265, upper bound: 0.0019213
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0018362, upper bound: 0.0017742
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0018539, upper bound: 0.0018589
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0018362, upper bound: 0.0017618
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0018539, upper bound: 0.0018444
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0018376, upper bound: 0.0017929
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0018563, upper bound: 0.0018741
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0018376, upper bound: 0.0017807
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.97
Output dim: 8, lower bound: -0.0018563, upper bound: 0.0018563

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0025080, 0.0049535, 0.0025727, 0.0049889, -0.0014344, 0.0012749
1: 0.0016846, 0.0020379, 0.0016940, 0.0020430, -0.0002072, 0.0001842
2: 0.0116212, 0.0129733, 0.0116017, 0.0129375, -0.0007049, 0.0007931
3: -0.0026613, -0.0012629, -0.0026815, -0.0012999, -0.0007290, 0.0008202
4: -0.0026698, -0.0011560, -0.0026297, -0.0011341, -0.0008879, 0.0007892
5: 0.0052073, 0.0066399, 0.0051866, 0.0066020, -0.0007469, 0.0008403
6: -0.0016393, 0.0040447, -0.0017215, 0.0038943, -0.0029633, 0.0033340
7: -0.0080653, -0.0003241, -0.0078604, -0.0002122, -0.0045406, 0.0040358
8: 0.9835325, 0.9889856, 0.9836769, 0.9890644, -0.0031985, 0.0028429
9: -0.0058891, -0.0009392, -0.0059607, -0.0010702, -0.0025806, 0.0029034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017927, upper bound: 0.0018332
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018544, upper bound: 0.0018492
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0025080, 0.0049535, 0.0026297, 0.0050633, -0.0016581, 0.0013674
1: 0.0016846, 0.0020379, 0.0017022, 0.0020538, -0.0002396, 0.0001975
2: 0.0116212, 0.0129733, 0.0115605, 0.0129060, -0.0007560, 0.0009167
3: -0.0026613, -0.0012629, -0.0027241, -0.0013325, -0.0007819, 0.0009481
4: -0.0026698, -0.0011560, -0.0025944, -0.0010880, -0.0010264, 0.0008464
5: 0.0052073, 0.0066399, 0.0051430, 0.0065686, -0.0008010, 0.0009713
6: -0.0016393, 0.0040447, -0.0018946, 0.0037618, -0.0031781, 0.0038540
7: -0.0080653, -0.0003241, -0.0076800, 0.0000235, -0.0052488, 0.0043283
8: 0.9835325, 0.9889856, 0.9838039, 0.9892305, -0.0036974, 0.0030490
9: -0.0058891, -0.0009392, -0.0061114, -0.0011855, -0.0027677, 0.0033562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017927, upper bound: 0.0018416
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018544, upper bound: 0.0018589
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0024860, 0.0049146, 0.0025739, 0.0049591, -0.0014865, 0.0012658
1: 0.0016815, 0.0020323, 0.0016942, 0.0020387, -0.0002148, 0.0001829
2: 0.0116427, 0.0129854, 0.0116181, 0.0129368, -0.0006998, 0.0008218
3: -0.0026390, -0.0012503, -0.0026645, -0.0013006, -0.0007238, 0.0008500
4: -0.0026834, -0.0011801, -0.0026290, -0.0011525, -0.0009201, 0.0007835
5: 0.0052301, 0.0066528, 0.0052040, 0.0066013, -0.0007415, 0.0008708
6: -0.0015488, 0.0040959, -0.0016523, 0.0038916, -0.0029420, 0.0034550
7: -0.0081349, -0.0004474, -0.0078567, -0.0003065, -0.0047053, 0.0040067
8: 0.9834835, 0.9888988, 0.9836794, 0.9889981, -0.0033145, 0.0028224
9: -0.0058103, -0.0008947, -0.0059004, -0.0010725, -0.0025620, 0.0030087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017927, upper bound: 0.0018152
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018546, upper bound: 0.0018326
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0024860, 0.0049146, 0.0026309, 0.0050347, -0.0016996, 0.0013581
1: 0.0016815, 0.0020323, 0.0017024, 0.0020497, -0.0002455, 0.0001962
2: 0.0116427, 0.0129854, 0.0115763, 0.0129053, -0.0007508, 0.0009397
3: -0.0026390, -0.0012503, -0.0027077, -0.0013332, -0.0007766, 0.0009719
4: -0.0026834, -0.0011801, -0.0025937, -0.0011057, -0.0010521, 0.0008407
5: 0.0052301, 0.0066528, 0.0051597, 0.0065678, -0.0007955, 0.0009956
6: -0.0015488, 0.0040959, -0.0018280, 0.0037590, -0.0031565, 0.0039504
7: -0.0081349, -0.0004474, -0.0076761, -0.0000671, -0.0053801, 0.0042989
8: 0.9834835, 0.9888988, 0.9838067, 0.9891666, -0.0037898, 0.0030282
9: -0.0058103, -0.0008947, -0.0060534, -0.0011880, -0.0027488, 0.0034402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017927, upper bound: 0.0018269
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018546, upper bound: 0.0018444
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0024936, 0.0049540, 0.0025380, 0.0050106, -0.0015882, 0.0012910
1: 0.0016825, 0.0020380, 0.0016890, 0.0020462, -0.0002294, 0.0001865
2: 0.0116210, 0.0129812, 0.0115896, 0.0129567, -0.0007137, 0.0008781
3: -0.0026615, -0.0012547, -0.0026939, -0.0012801, -0.0007382, 0.0009081
4: -0.0026787, -0.0011557, -0.0026512, -0.0011206, -0.0009831, 0.0007991
5: 0.0052070, 0.0066483, 0.0051739, 0.0066223, -0.0007562, 0.0009303
6: -0.0016403, 0.0040783, -0.0017720, 0.0039749, -0.0030006, 0.0036913
7: -0.0081109, -0.0003227, -0.0079702, -0.0001435, -0.0050273, 0.0040865
8: 0.9835003, 0.9889866, 0.9835994, 0.9891128, -0.0035413, 0.0028786
9: -0.0058900, -0.0009100, -0.0060046, -0.0010000, -0.0026130, 0.0032146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018498
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018566, upper bound: 0.0018677
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0024936, 0.0049540, 0.0025906, 0.0050918, -0.0018072, 0.0013827
1: 0.0016825, 0.0020380, 0.0016966, 0.0020579, -0.0002611, 0.0001998
2: 0.0116210, 0.0129812, 0.0115447, 0.0129276, -0.0007645, 0.0009991
3: -0.0026615, -0.0012547, -0.0027404, -0.0013102, -0.0007907, 0.0010334
4: -0.0026787, -0.0011557, -0.0026186, -0.0010704, -0.0011187, 0.0008559
5: 0.0052070, 0.0066483, 0.0051263, 0.0065915, -0.0008100, 0.0010586
6: -0.0016403, 0.0040783, -0.0019608, 0.0038526, -0.0032138, 0.0042004
7: -0.0081109, -0.0003227, -0.0078037, 0.0001137, -0.0057205, 0.0043770
8: 0.9835003, 0.9889866, 0.9837168, 0.9892939, -0.0040297, 0.0030832
9: -0.0058900, -0.0009100, -0.0061690, -0.0011065, -0.0027988, 0.0036579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018558
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018566, upper bound: 0.0018741
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0024749, 0.0049151, 0.0025393, 0.0049792, -0.0016455, 0.0012816
1: 0.0016799, 0.0020324, 0.0016892, 0.0020416, -0.0002377, 0.0001852
2: 0.0116425, 0.0129916, 0.0116070, 0.0129560, -0.0007086, 0.0009098
3: -0.0026393, -0.0012440, -0.0026759, -0.0012808, -0.0007328, 0.0009409
4: -0.0026903, -0.0011798, -0.0026504, -0.0011401, -0.0010186, 0.0007933
5: 0.0052298, 0.0066593, 0.0051923, 0.0066215, -0.0007508, 0.0009639
6: -0.0015500, 0.0041217, -0.0016990, 0.0039720, -0.0029788, 0.0038246
7: -0.0081701, -0.0004458, -0.0079663, -0.0002428, -0.0052088, 0.0040569
8: 0.9834586, 0.9888998, 0.9836022, 0.9890428, -0.0036692, 0.0028578
9: -0.0058113, -0.0008722, -0.0059411, -0.0010025, -0.0025941, 0.0033307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018284
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018567, upper bound: 0.0018461
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0024749, 0.0049151, 0.0025920, 0.0050634, -0.0018545, 0.0013733
1: 0.0016799, 0.0020324, 0.0016968, 0.0020538, -0.0002679, 0.0001984
2: 0.0116425, 0.0129916, 0.0115604, 0.0129268, -0.0007592, 0.0010253
3: -0.0026393, -0.0012440, -0.0027241, -0.0013109, -0.0007853, 0.0010604
4: -0.0026903, -0.0011798, -0.0026178, -0.0010879, -0.0011480, 0.0008501
5: 0.0052298, 0.0066593, 0.0051429, 0.0065907, -0.0008045, 0.0010864
6: -0.0015500, 0.0041217, -0.0018948, 0.0038496, -0.0031919, 0.0043104
7: -0.0081701, -0.0004458, -0.0077995, 0.0000238, -0.0058704, 0.0043471
8: 0.9834586, 0.9888998, 0.9837196, 0.9892306, -0.0041353, 0.0030622
9: -0.0058113, -0.0008722, -0.0061116, -0.0011091, -0.0027796, 0.0037537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018376
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018567, upper bound: 0.0018563
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026409, 0.0050279, 0.0027221, 0.0050397, -0.0013380, 0.0014589
1: 0.0017038, 0.0020487, 0.0017156, 0.0020504, -0.0001933, 0.0002108
2: 0.0115801, 0.0128998, 0.0115736, 0.0128549, -0.0008066, 0.0007397
3: -0.0027038, -0.0013389, -0.0027105, -0.0013853, -0.0008342, 0.0007651
4: -0.0025875, -0.0011099, -0.0025373, -0.0011026, -0.0008282, 0.0009031
5: 0.0051637, 0.0065620, 0.0051568, 0.0065145, -0.0008546, 0.0007838
6: -0.0018122, 0.0037359, -0.0018396, 0.0035471, -0.0033910, 0.0031098
7: -0.0076447, -0.0000887, -0.0073876, -0.0000514, -0.0042352, 0.0046182
8: 0.9838289, 0.9891515, 0.9840099, 0.9891777, -0.0029834, 0.0032532
9: -0.0060397, -0.0012081, -0.0060635, -0.0013725, -0.0029530, 0.0027081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017742
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017742
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0025698, 0.0050291, 0.0025984, 0.0049912, -0.0014363, 0.0014944
1: 0.0016936, 0.0020489, 0.0016977, 0.0020434, -0.0002075, 0.0002159
2: 0.0115794, 0.0129391, 0.0116004, 0.0129233, -0.0008262, 0.0007941
3: -0.0027045, -0.0012982, -0.0026828, -0.0013146, -0.0008545, 0.0008213
4: -0.0026315, -0.0011092, -0.0026138, -0.0011327, -0.0008891, 0.0009250
5: 0.0051630, 0.0066037, 0.0051852, 0.0065869, -0.0008754, 0.0008414
6: -0.0018150, 0.0039011, -0.0017269, 0.0038347, -0.0034734, 0.0033383
7: -0.0078697, -0.0000848, -0.0077792, -0.0002048, -0.0045464, 0.0047304
8: 0.9836702, 0.9891540, 0.9837341, 0.9890696, -0.0032026, 0.0033322
9: -0.0060421, -0.0010642, -0.0059654, -0.0011221, -0.0030248, 0.0029071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018416
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018589
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026222, 0.0049894, 0.0027255, 0.0050116, -0.0013893, 0.0014081
1: 0.0017011, 0.0020431, 0.0017161, 0.0020463, -0.0002007, 0.0002034
2: 0.0116014, 0.0129101, 0.0115891, 0.0128530, -0.0007785, 0.0007681
3: -0.0026818, -0.0013282, -0.0026945, -0.0013873, -0.0008052, 0.0007944
4: -0.0025991, -0.0011338, -0.0025351, -0.0011200, -0.0008600, 0.0008717
5: 0.0051863, 0.0065730, 0.0051733, 0.0065125, -0.0008249, 0.0008139
6: -0.0017226, 0.0037793, -0.0017743, 0.0035392, -0.0032729, 0.0032291
7: -0.0077038, -0.0002106, -0.0073768, -0.0001403, -0.0043978, 0.0044574
8: 0.9837871, 0.9890654, 0.9840175, 0.9891151, -0.0030979, 0.0031399
9: -0.0059617, -0.0011703, -0.0060066, -0.0013794, -0.0028502, 0.0028121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017618
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017618
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025479, 0.0049908, 0.0025996, 0.0049615, -0.0014630, 0.0014492
1: 0.0016904, 0.0020433, 0.0016979, 0.0020391, -0.0002114, 0.0002094
2: 0.0116006, 0.0129512, 0.0116168, 0.0129226, -0.0008012, 0.0008089
3: -0.0026826, -0.0012857, -0.0026659, -0.0013153, -0.0008286, 0.0008366
4: -0.0026451, -0.0011329, -0.0026131, -0.0011510, -0.0009057, 0.0008971
5: 0.0051854, 0.0066165, 0.0052026, 0.0065862, -0.0008489, 0.0008571
6: -0.0017260, 0.0039519, -0.0016580, 0.0038319, -0.0033683, 0.0034005
7: -0.0079389, -0.0002060, -0.0077754, -0.0002987, -0.0046312, 0.0045873
8: 0.9836215, 0.9890687, 0.9837368, 0.9890034, -0.0032623, 0.0032314
9: -0.0059646, -0.0010200, -0.0059054, -0.0011246, -0.0029332, 0.0029613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018269
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018444
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026230, 0.0050285, 0.0026785, 0.0050636, -0.0014717, 0.0014728
1: 0.0017012, 0.0020488, 0.0017093, 0.0020538, -0.0002126, 0.0002128
2: 0.0115797, 0.0129097, 0.0115603, 0.0128790, -0.0008143, 0.0008136
3: -0.0027041, -0.0013287, -0.0027242, -0.0013604, -0.0008422, 0.0008415
4: -0.0025986, -0.0011096, -0.0025642, -0.0010878, -0.0009110, 0.0009117
5: 0.0051634, 0.0065725, 0.0051428, 0.0065400, -0.0008628, 0.0008621
6: -0.0018136, 0.0037774, -0.0018952, 0.0036483, -0.0034233, 0.0034205
7: -0.0077013, -0.0000867, -0.0075254, 0.0000243, -0.0046585, 0.0046622
8: 0.9837890, 0.9891527, 0.9839128, 0.9892311, -0.0032815, 0.0032842
9: -0.0060409, -0.0011719, -0.0061119, -0.0012844, -0.0029812, 0.0029787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017919
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017766
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0025540, 0.0050297, 0.0025578, 0.0050131, -0.0015566, 0.0015075
1: 0.0016913, 0.0020489, 0.0016918, 0.0020466, -0.0002249, 0.0002178
2: 0.0115791, 0.0129478, 0.0115882, 0.0129457, -0.0008335, 0.0008606
3: -0.0027048, -0.0012892, -0.0026954, -0.0012914, -0.0008620, 0.0008901
4: -0.0026413, -0.0011088, -0.0026390, -0.0011191, -0.0009635, 0.0009332
5: 0.0051627, 0.0066129, 0.0051724, 0.0066107, -0.0008831, 0.0009118
6: -0.0018164, 0.0039378, -0.0017779, 0.0039291, -0.0035039, 0.0036179
7: -0.0079197, -0.0000829, -0.0079077, -0.0001353, -0.0049272, 0.0047720
8: 0.9836351, 0.9891555, 0.9836435, 0.9891185, -0.0034708, 0.0033615
9: -0.0060433, -0.0010323, -0.0060098, -0.0010399, -0.0030513, 0.0031506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018558
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018741
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026074, 0.0049899, 0.0026821, 0.0050358, -0.0015580, 0.0014226
1: 0.0016990, 0.0020432, 0.0017098, 0.0020498, -0.0002251, 0.0002055
2: 0.0116011, 0.0129183, 0.0115757, 0.0128770, -0.0007865, 0.0008614
3: -0.0026821, -0.0013198, -0.0027083, -0.0013625, -0.0008135, 0.0008909
4: -0.0026082, -0.0011334, -0.0025620, -0.0011050, -0.0009644, 0.0008806
5: 0.0051860, 0.0065816, 0.0051591, 0.0065379, -0.0008334, 0.0009127
6: -0.0017240, 0.0038136, -0.0018307, 0.0036401, -0.0033066, 0.0036212
7: -0.0077505, -0.0002088, -0.0075142, -0.0000635, -0.0049318, 0.0045033
8: 0.9837543, 0.9890668, 0.9839208, 0.9891691, -0.0034740, 0.0031722
9: -0.0059628, -0.0011405, -0.0060557, -0.0012916, -0.0028795, 0.0031535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017799
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017639
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025347, 0.0049914, 0.0025590, 0.0049818, -0.0016215, 0.0014630
1: 0.0016885, 0.0020434, 0.0016920, 0.0020420, -0.0002343, 0.0002114
2: 0.0116003, 0.0129585, 0.0116055, 0.0129451, -0.0008089, 0.0008965
3: -0.0026829, -0.0012782, -0.0026775, -0.0012921, -0.0008366, 0.0009272
4: -0.0026533, -0.0011325, -0.0026382, -0.0011384, -0.0010037, 0.0009056
5: 0.0051851, 0.0066242, 0.0051907, 0.0066100, -0.0008570, 0.0009499
6: -0.0017274, 0.0039827, -0.0017052, 0.0039261, -0.0034005, 0.0037688
7: -0.0079808, -0.0002042, -0.0079038, -0.0002344, -0.0051328, 0.0046311
8: 0.9835920, 0.9890701, 0.9836462, 0.9890487, -0.0036156, 0.0032623
9: -0.0059658, -0.0009932, -0.0059464, -0.0010425, -0.0029613, 0.0032820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018376
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018563
time: 0.97 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.28 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017927, upper bound: 0.0018332
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018544, upper bound: 0.0018492
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017927, upper bound: 0.0018416
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018544, upper bound: 0.0018589
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017927, upper bound: 0.0018152
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018546, upper bound: 0.0018326
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017927, upper bound: 0.0018269
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018546, upper bound: 0.0018444
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018498
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018566, upper bound: 0.0018677
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018558
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018566, upper bound: 0.0018741
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018284
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018567, upper bound: 0.0018461
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018376
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018567, upper bound: 0.0018563
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017742
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017742
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018416
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018589
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017618
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017618
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018269
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018444
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017919
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017766
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018558
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018741
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017799
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017639
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018376
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018563

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026680, 0.0050002, 0.0026487, 0.0049875, -0.0012506, 0.0011662
1: 0.0017078, 0.0020447, 0.0017050, 0.0020429, -0.0001807, 0.0001685
2: 0.0115954, 0.0128848, 0.0116024, 0.0128954, -0.0006448, 0.0006914
3: -0.0026880, -0.0013544, -0.0026807, -0.0013434, -0.0006669, 0.0007151
4: -0.0025707, -0.0011271, -0.0025827, -0.0011349, -0.0007741, 0.0007219
5: 0.0051799, 0.0065461, 0.0051874, 0.0065574, -0.0006832, 0.0007326
6: -0.0017479, 0.0036727, -0.0017183, 0.0037176, -0.0027107, 0.0029067
7: -0.0075587, -0.0001763, -0.0076197, -0.0002165, -0.0039587, 0.0036917
8: 0.9838895, 0.9890897, 0.9838464, 0.9890614, -0.0027886, 0.0026005
9: -0.0059836, -0.0012631, -0.0059579, -0.0012241, -0.0023606, 0.0025313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017947, upper bound: 0.0018495
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017947, upper bound: 0.0018495
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025463, 0.0049528, 0.0025793, 0.0049887, -0.0012409, 0.0012707
1: 0.0016902, 0.0020378, 0.0016949, 0.0020430, -0.0001793, 0.0001836
2: 0.0116216, 0.0129521, 0.0116017, 0.0129338, -0.0007026, 0.0006861
3: -0.0026609, -0.0012848, -0.0026814, -0.0013037, -0.0007266, 0.0007096
4: -0.0026461, -0.0011564, -0.0026256, -0.0011342, -0.0007681, 0.0007866
5: 0.0052077, 0.0066174, 0.0051867, 0.0065981, -0.0007444, 0.0007269
6: -0.0016377, 0.0039557, -0.0017212, 0.0038789, -0.0029535, 0.0028842
7: -0.0079440, -0.0003262, -0.0078395, -0.0002126, -0.0039281, 0.0040224
8: 0.9836179, 0.9889840, 0.9836915, 0.9890641, -0.0027670, 0.0028335
9: -0.0058877, -0.0010167, -0.0059604, -0.0010836, -0.0025720, 0.0025117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018393, upper bound: 0.0017954
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018393, upper bound: 0.0018649
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026680, 0.0050002, 0.0027095, 0.0050621, -0.0014744, 0.0012650
1: 0.0017078, 0.0020447, 0.0017137, 0.0020536, -0.0002130, 0.0001828
2: 0.0115954, 0.0128848, 0.0115612, 0.0128619, -0.0006994, 0.0008152
3: -0.0026880, -0.0013544, -0.0027234, -0.0013781, -0.0007233, 0.0008431
4: -0.0025707, -0.0011271, -0.0025451, -0.0010888, -0.0009127, 0.0007831
5: 0.0051799, 0.0065461, 0.0051437, 0.0065218, -0.0007410, 0.0008637
6: -0.0017479, 0.0036727, -0.0018917, 0.0035764, -0.0029402, 0.0034269
7: -0.0075587, -0.0001763, -0.0074274, 0.0000196, -0.0046672, 0.0040044
8: 0.9838895, 0.9890897, 0.9839818, 0.9892277, -0.0032877, 0.0028208
9: -0.0059836, -0.0012631, -0.0061089, -0.0013471, -0.0025605, 0.0029843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017925, upper bound: 0.0018416
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017925, upper bound: 0.0018416
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025463, 0.0049528, 0.0026360, 0.0050632, -0.0015118, 0.0013614
1: 0.0016902, 0.0020378, 0.0017031, 0.0020538, -0.0002184, 0.0001967
2: 0.0116216, 0.0129521, 0.0115605, 0.0129025, -0.0007527, 0.0008358
3: -0.0026609, -0.0012848, -0.0027240, -0.0013361, -0.0007785, 0.0008644
4: -0.0026461, -0.0011564, -0.0025905, -0.0010881, -0.0009358, 0.0008427
5: 0.0052077, 0.0066174, 0.0051430, 0.0065649, -0.0007975, 0.0008856
6: -0.0016377, 0.0039557, -0.0018943, 0.0037472, -0.0031643, 0.0035137
7: -0.0079440, -0.0003262, -0.0076601, 0.0000232, -0.0047854, 0.0043096
8: 0.9836179, 0.9889840, 0.9838180, 0.9892302, -0.0033709, 0.0030357
9: -0.0058877, -0.0010167, -0.0061112, -0.0011983, -0.0027556, 0.0030599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018374, upper bound: 0.0017742
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018374, upper bound: 0.0018589
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026468, 0.0049649, 0.0026499, 0.0049578, -0.0013115, 0.0011630
1: 0.0017047, 0.0020396, 0.0017051, 0.0020386, -0.0001895, 0.0001680
2: 0.0116149, 0.0128965, 0.0116188, 0.0128948, -0.0006430, 0.0007251
3: -0.0026678, -0.0013423, -0.0026637, -0.0013441, -0.0006650, 0.0007499
4: -0.0025839, -0.0011489, -0.0025819, -0.0011533, -0.0008119, 0.0007199
5: 0.0052006, 0.0065586, 0.0052048, 0.0065567, -0.0006813, 0.0007683
6: -0.0016657, 0.0037221, -0.0016493, 0.0037148, -0.0027032, 0.0030483
7: -0.0076260, -0.0002881, -0.0076159, -0.0003106, -0.0041516, 0.0036815
8: 0.9838420, 0.9890109, 0.9838490, 0.9889952, -0.0029245, 0.0025933
9: -0.0059121, -0.0012201, -0.0058978, -0.0012265, -0.0023541, 0.0026546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017948, upper bound: 0.0017822
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017948, upper bound: 0.0018291
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025258, 0.0049137, 0.0025805, 0.0049590, -0.0013132, 0.0012614
1: 0.0016872, 0.0020322, 0.0016951, 0.0020387, -0.0001897, 0.0001822
2: 0.0116432, 0.0129634, 0.0116182, 0.0129332, -0.0006974, 0.0007260
3: -0.0026385, -0.0012731, -0.0026644, -0.0013044, -0.0007213, 0.0007509
4: -0.0026588, -0.0011806, -0.0026249, -0.0011526, -0.0008129, 0.0007808
5: 0.0052306, 0.0066294, 0.0052041, 0.0065974, -0.0007389, 0.0007693
6: -0.0015469, 0.0040033, -0.0016520, 0.0038762, -0.0029318, 0.0030522
7: -0.0080089, -0.0004500, -0.0078358, -0.0003068, -0.0041569, 0.0039929
8: 0.9835722, 0.9888968, 0.9836942, 0.9889978, -0.0029282, 0.0028127
9: -0.0058086, -0.0009753, -0.0059002, -0.0010859, -0.0025532, 0.0026580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018396, upper bound: 0.0017822
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018396, upper bound: 0.0018447
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026468, 0.0049649, 0.0027108, 0.0050335, -0.0015248, 0.0012617
1: 0.0017047, 0.0020396, 0.0017139, 0.0020495, -0.0002203, 0.0001823
2: 0.0116149, 0.0128965, 0.0115770, 0.0128612, -0.0006976, 0.0008430
3: -0.0026678, -0.0013423, -0.0027070, -0.0013789, -0.0007214, 0.0008719
4: -0.0025839, -0.0011489, -0.0025443, -0.0011065, -0.0009439, 0.0007810
5: 0.0052006, 0.0065586, 0.0051605, 0.0065211, -0.0007391, 0.0008932
6: -0.0016657, 0.0037221, -0.0018251, 0.0035734, -0.0029325, 0.0035441
7: -0.0076260, -0.0002881, -0.0074234, -0.0000710, -0.0048267, 0.0039938
8: 0.9838420, 0.9890109, 0.9839846, 0.9891639, -0.0034001, 0.0028133
9: -0.0059121, -0.0012201, -0.0060509, -0.0013496, -0.0025538, 0.0030864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017926, upper bound: 0.0017618
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017926, upper bound: 0.0018269
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025258, 0.0049137, 0.0026372, 0.0050346, -0.0015680, 0.0013520
1: 0.0016872, 0.0020322, 0.0017033, 0.0020497, -0.0002265, 0.0001953
2: 0.0116432, 0.0129634, 0.0115764, 0.0129018, -0.0007475, 0.0008669
3: -0.0026385, -0.0012731, -0.0027076, -0.0013368, -0.0007731, 0.0008966
4: -0.0026588, -0.0011806, -0.0025898, -0.0011058, -0.0009706, 0.0008369
5: 0.0052306, 0.0066294, 0.0051598, 0.0065642, -0.0007920, 0.0009185
6: -0.0015469, 0.0040033, -0.0018277, 0.0037443, -0.0031423, 0.0036444
7: -0.0080089, -0.0004500, -0.0076562, -0.0000675, -0.0049633, 0.0042796
8: 0.9835722, 0.9888968, 0.9838207, 0.9891663, -0.0034963, 0.0030146
9: -0.0058086, -0.0009753, -0.0060532, -0.0012008, -0.0027365, 0.0031737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018378, upper bound: 0.0017618
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018378, upper bound: 0.0018444
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026539, 0.0050006, 0.0026153, 0.0050093, -0.0014053, 0.0011816
1: 0.0017057, 0.0020447, 0.0017001, 0.0020460, -0.0002030, 0.0001707
2: 0.0115952, 0.0128926, 0.0115904, 0.0129139, -0.0006533, 0.0007770
3: -0.0026882, -0.0013463, -0.0026932, -0.0013243, -0.0006756, 0.0008036
4: -0.0025795, -0.0011269, -0.0026034, -0.0011215, -0.0008699, 0.0007314
5: 0.0051797, 0.0065544, 0.0051746, 0.0065770, -0.0006922, 0.0008232
6: -0.0017487, 0.0037056, -0.0017689, 0.0037953, -0.0027463, 0.0032664
7: -0.0076034, -0.0001752, -0.0077255, -0.0001476, -0.0044486, 0.0037402
8: 0.9838579, 0.9890904, 0.9837719, 0.9891099, -0.0031337, 0.0026347
9: -0.0059843, -0.0012345, -0.0060020, -0.0011564, -0.0023916, 0.0028445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017947, upper bound: 0.0018624
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017947, upper bound: 0.0018624
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025314, 0.0049533, 0.0025447, 0.0050105, -0.0014137, 0.0012867
1: 0.0016880, 0.0020379, 0.0016899, 0.0020462, -0.0002042, 0.0001859
2: 0.0116213, 0.0129603, 0.0115897, 0.0129530, -0.0007114, 0.0007816
3: -0.0026611, -0.0012763, -0.0026938, -0.0012839, -0.0007357, 0.0008084
4: -0.0026553, -0.0011561, -0.0026471, -0.0011207, -0.0008751, 0.0007965
5: 0.0052074, 0.0066262, 0.0051739, 0.0066184, -0.0007537, 0.0008281
6: -0.0016388, 0.0039904, -0.0017717, 0.0039594, -0.0029906, 0.0032858
7: -0.0079913, -0.0003249, -0.0079491, -0.0001438, -0.0044749, 0.0040729
8: 0.9835846, 0.9889851, 0.9836144, 0.9891126, -0.0031522, 0.0028690
9: -0.0058886, -0.0009865, -0.0060044, -0.0010135, -0.0026043, 0.0028614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018395, upper bound: 0.0018059
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018395, upper bound: 0.0018792
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026539, 0.0050006, 0.0026681, 0.0050905, -0.0016244, 0.0012793
1: 0.0017057, 0.0020447, 0.0017078, 0.0020577, -0.0002347, 0.0001848
2: 0.0115952, 0.0128926, 0.0115454, 0.0128848, -0.0007073, 0.0008981
3: -0.0026882, -0.0013463, -0.0027396, -0.0013544, -0.0007315, 0.0009288
4: -0.0025795, -0.0011269, -0.0025707, -0.0010712, -0.0010055, 0.0007919
5: 0.0051797, 0.0065544, 0.0051270, 0.0065461, -0.0007494, 0.0009516
6: -0.0017487, 0.0037056, -0.0019578, 0.0036727, -0.0029734, 0.0037755
7: -0.0076034, -0.0001752, -0.0075586, 0.0001097, -0.0051419, 0.0040495
8: 0.9838579, 0.9890904, 0.9838895, 0.9892911, -0.0036221, 0.0028525
9: -0.0059843, -0.0012345, -0.0061665, -0.0012632, -0.0025894, 0.0032879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017931, upper bound: 0.0018558
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017931, upper bound: 0.0018558
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025314, 0.0049533, 0.0025977, 0.0050917, -0.0016574, 0.0013768
1: 0.0016880, 0.0020379, 0.0016976, 0.0020579, -0.0002394, 0.0001989
2: 0.0116213, 0.0129603, 0.0115448, 0.0129237, -0.0007612, 0.0009163
3: -0.0026611, -0.0012763, -0.0027403, -0.0013142, -0.0007873, 0.0009477
4: -0.0026553, -0.0011561, -0.0026143, -0.0010704, -0.0010259, 0.0008523
5: 0.0052074, 0.0066262, 0.0051263, 0.0065873, -0.0008065, 0.0009709
6: -0.0016388, 0.0039904, -0.0019605, 0.0038363, -0.0032001, 0.0038522
7: -0.0079913, -0.0003249, -0.0077814, 0.0001133, -0.0052464, 0.0043583
8: 0.9835846, 0.9889851, 0.9837325, 0.9892937, -0.0036957, 0.0030701
9: -0.0058886, -0.0009865, -0.0061688, -0.0011207, -0.0027868, 0.0033547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018383, upper bound: 0.0017929
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018383, upper bound: 0.0018741
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026350, 0.0049653, 0.0026166, 0.0049779, -0.0014679, 0.0011782
1: 0.0017030, 0.0020396, 0.0017003, 0.0020415, -0.0002121, 0.0001702
2: 0.0116147, 0.0129030, 0.0116077, 0.0129132, -0.0006514, 0.0008116
3: -0.0026680, -0.0013355, -0.0026752, -0.0013250, -0.0006737, 0.0008394
4: -0.0025912, -0.0011487, -0.0026026, -0.0011409, -0.0009087, 0.0007293
5: 0.0052004, 0.0065655, 0.0051930, 0.0065763, -0.0006902, 0.0008599
6: -0.0016667, 0.0037495, -0.0016960, 0.0037924, -0.0027384, 0.0034119
7: -0.0076632, -0.0002868, -0.0077216, -0.0002469, -0.0046467, 0.0037294
8: 0.9838157, 0.9890119, 0.9837746, 0.9890399, -0.0032732, 0.0026271
9: -0.0059130, -0.0011963, -0.0059385, -0.0011590, -0.0023847, 0.0029712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017948, upper bound: 0.0017948
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017948, upper bound: 0.0018397
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025144, 0.0049142, 0.0025460, 0.0049791, -0.0014875, 0.0012772
1: 0.0016856, 0.0020323, 0.0016901, 0.0020416, -0.0002149, 0.0001845
2: 0.0116429, 0.0129697, 0.0116071, 0.0129523, -0.0007061, 0.0008224
3: -0.0026388, -0.0012666, -0.0026759, -0.0012846, -0.0007303, 0.0008506
4: -0.0026658, -0.0011803, -0.0026463, -0.0011402, -0.0009208, 0.0007906
5: 0.0052303, 0.0066361, 0.0051923, 0.0066176, -0.0007482, 0.0008714
6: -0.0015481, 0.0040297, -0.0016987, 0.0039565, -0.0029685, 0.0034574
7: -0.0080449, -0.0004484, -0.0079451, -0.0002432, -0.0047086, 0.0040428
8: 0.9835469, 0.9888980, 0.9836172, 0.9890425, -0.0033168, 0.0028478
9: -0.0058096, -0.0009522, -0.0059408, -0.0010160, -0.0025851, 0.0030108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018397, upper bound: 0.0017948
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018397, upper bound: 0.0018578
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026350, 0.0049653, 0.0026694, 0.0050622, -0.0016770, 0.0012758
1: 0.0017030, 0.0020396, 0.0017079, 0.0020536, -0.0002423, 0.0001843
2: 0.0116147, 0.0129030, 0.0115611, 0.0128840, -0.0007053, 0.0009272
3: -0.0026680, -0.0013355, -0.0027234, -0.0013552, -0.0007295, 0.0009589
4: -0.0025912, -0.0011487, -0.0025699, -0.0010887, -0.0010381, 0.0007897
5: 0.0052004, 0.0065655, 0.0051436, 0.0065453, -0.0007473, 0.0009824
6: -0.0016667, 0.0037495, -0.0018919, 0.0036696, -0.0029652, 0.0038979
7: -0.0076632, -0.0002868, -0.0075544, 0.0000199, -0.0053086, 0.0040384
8: 0.9838157, 0.9890119, 0.9838924, 0.9892279, -0.0037395, 0.0028447
9: -0.0059130, -0.0011963, -0.0061091, -0.0012658, -0.0025823, 0.0033944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0017807
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018376
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025144, 0.0049142, 0.0025990, 0.0050633, -0.0017157, 0.0013672
1: 0.0016856, 0.0020323, 0.0016978, 0.0020538, -0.0002479, 0.0001975
2: 0.0116429, 0.0129697, 0.0115605, 0.0129230, -0.0007559, 0.0009486
3: -0.0026388, -0.0012666, -0.0027241, -0.0013149, -0.0007818, 0.0009811
4: -0.0026658, -0.0011803, -0.0026135, -0.0010880, -0.0010621, 0.0008463
5: 0.0052303, 0.0066361, 0.0051430, 0.0065866, -0.0008009, 0.0010051
6: -0.0015481, 0.0040297, -0.0018945, 0.0038333, -0.0031778, 0.0039878
7: -0.0080449, -0.0004484, -0.0077773, 0.0000235, -0.0054311, 0.0043279
8: 0.9835469, 0.9888980, 0.9837354, 0.9892304, -0.0038258, 0.0030486
9: -0.0058096, -0.0009522, -0.0061114, -0.0011233, -0.0027674, 0.0034728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018384, upper bound: 0.0017807
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018384, upper bound: 0.0018563
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027149, 0.0050254, 0.0027221, 0.0050397, -0.0012105, 0.0014573
1: 0.0017145, 0.0020483, 0.0017156, 0.0020504, -0.0001749, 0.0002105
2: 0.0115814, 0.0128589, 0.0115736, 0.0128549, -0.0008057, 0.0006692
3: -0.0027024, -0.0013812, -0.0027105, -0.0013853, -0.0008333, 0.0006922
4: -0.0025417, -0.0011114, -0.0025373, -0.0011026, -0.0007493, 0.0009021
5: 0.0051652, 0.0065187, 0.0051568, 0.0065145, -0.0008537, 0.0007091
6: -0.0018065, 0.0035639, -0.0018396, 0.0035471, -0.0033871, 0.0028135
7: -0.0074105, -0.0000964, -0.0073876, -0.0000514, -0.0038317, 0.0046129
8: 0.9839938, 0.9891460, 0.9840099, 0.9891777, -0.0026992, 0.0032494
9: -0.0060347, -0.0013579, -0.0060635, -0.0013725, -0.0029496, 0.0024501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017706
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017706
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026735, 0.0050523, 0.0027221, 0.0050397, -0.0013284, 0.0015544
1: 0.0017085, 0.0020522, 0.0017156, 0.0020504, -0.0001919, 0.0002246
2: 0.0115666, 0.0128817, 0.0115736, 0.0128549, -0.0008594, 0.0007344
3: -0.0027178, -0.0013576, -0.0027105, -0.0013853, -0.0008888, 0.0007596
4: -0.0025673, -0.0010948, -0.0025373, -0.0011026, -0.0008223, 0.0009622
5: 0.0051494, 0.0065429, 0.0051568, 0.0065145, -0.0009106, 0.0007782
6: -0.0018690, 0.0036600, -0.0018396, 0.0035471, -0.0036130, 0.0030875
7: -0.0075413, -0.0000112, -0.0073876, -0.0000514, -0.0042049, 0.0049205
8: 0.9839016, 0.9892060, 0.9840099, 0.9891777, -0.0029620, 0.0034661
9: -0.0060892, -0.0012742, -0.0060635, -0.0013725, -0.0031463, 0.0026887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017706
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017706
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027228, 0.0050705, 0.0025984, 0.0049912, -0.0012554, 0.0016615
1: 0.0017157, 0.0020548, 0.0016977, 0.0020434, -0.0001814, 0.0002400
2: 0.0115565, 0.0128545, 0.0116004, 0.0129233, -0.0009186, 0.0006941
3: -0.0027281, -0.0013858, -0.0026828, -0.0013146, -0.0009501, 0.0007179
4: -0.0025368, -0.0010836, -0.0026138, -0.0011327, -0.0007771, 0.0010285
5: 0.0051388, 0.0065140, 0.0051852, 0.0065869, -0.0009733, 0.0007354
6: -0.0019112, 0.0035454, -0.0017269, 0.0038347, -0.0038618, 0.0029180
7: -0.0073852, 0.0000461, -0.0077792, -0.0002048, -0.0039741, 0.0052594
8: 0.9840117, 0.9892463, 0.9837341, 0.9890696, -0.0027994, 0.0037048
9: -0.0061258, -0.0013740, -0.0059654, -0.0011221, -0.0033630, 0.0025411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018332
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018332
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026015, 0.0050285, 0.0025984, 0.0049912, -0.0012474, 0.0014937
1: 0.0016981, 0.0020488, 0.0016977, 0.0020434, -0.0001802, 0.0002158
2: 0.0115797, 0.0129216, 0.0116004, 0.0129233, -0.0008258, 0.0006896
3: -0.0027042, -0.0013164, -0.0026828, -0.0013146, -0.0008541, 0.0007133
4: -0.0026119, -0.0011095, -0.0026138, -0.0011327, -0.0007721, 0.0009246
5: 0.0051633, 0.0065851, 0.0051852, 0.0065869, -0.0008750, 0.0007307
6: -0.0018137, 0.0038274, -0.0017269, 0.0038347, -0.0034718, 0.0028992
7: -0.0077693, -0.0000866, -0.0077792, -0.0002048, -0.0039485, 0.0047283
8: 0.9837410, 0.9891529, 0.9837341, 0.9890696, -0.0027814, 0.0033307
9: -0.0060410, -0.0011285, -0.0059654, -0.0011221, -0.0030234, 0.0025248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0017921
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0017921
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026952, 0.0049869, 0.0027255, 0.0050116, -0.0013008, 0.0014065
1: 0.0017117, 0.0020428, 0.0017161, 0.0020463, -0.0001879, 0.0002032
2: 0.0116027, 0.0128697, 0.0115891, 0.0128530, -0.0007776, 0.0007192
3: -0.0026804, -0.0013700, -0.0026945, -0.0013873, -0.0008043, 0.0007438
4: -0.0025539, -0.0011353, -0.0025351, -0.0011200, -0.0008052, 0.0008707
5: 0.0051877, 0.0065302, 0.0051733, 0.0065125, -0.0008239, 0.0007620
6: -0.0017169, 0.0036095, -0.0017743, 0.0035392, -0.0032692, 0.0030234
7: -0.0074726, -0.0002184, -0.0073768, -0.0001403, -0.0041177, 0.0044523
8: 0.9839500, 0.9890600, 0.9840175, 0.9891151, -0.0029006, 0.0031363
9: -0.0059567, -0.0013182, -0.0060066, -0.0013794, -0.0028469, 0.0026329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017581
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017581
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026581, 0.0050153, 0.0027255, 0.0050116, -0.0014222, 0.0015084
1: 0.0017063, 0.0020469, 0.0017161, 0.0020463, -0.0002055, 0.0002179
2: 0.0115870, 0.0128903, 0.0115891, 0.0128530, -0.0008339, 0.0007863
3: -0.0026966, -0.0013487, -0.0026945, -0.0013873, -0.0008625, 0.0008132
4: -0.0025769, -0.0011177, -0.0025351, -0.0011200, -0.0008804, 0.0009337
5: 0.0051711, 0.0065520, 0.0051733, 0.0065125, -0.0008836, 0.0008331
6: -0.0017830, 0.0036959, -0.0017743, 0.0035392, -0.0035058, 0.0033056
7: -0.0075902, -0.0001284, -0.0073768, -0.0001403, -0.0045019, 0.0047746
8: 0.9838672, 0.9891235, 0.9840175, 0.9891151, -0.0031712, 0.0033633
9: -0.0060143, -0.0012430, -0.0060066, -0.0013794, -0.0030530, 0.0028786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017581
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017581
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027065, 0.0050339, 0.0025996, 0.0049615, -0.0012872, 0.0016210
1: 0.0017133, 0.0020495, 0.0016979, 0.0020391, -0.0001860, 0.0002342
2: 0.0115768, 0.0128635, 0.0116168, 0.0129226, -0.0008962, 0.0007116
3: -0.0027072, -0.0013764, -0.0026659, -0.0013153, -0.0009269, 0.0007360
4: -0.0025469, -0.0011062, -0.0026131, -0.0011510, -0.0007968, 0.0010034
5: 0.0051602, 0.0065236, 0.0052026, 0.0065862, -0.0009496, 0.0007540
6: -0.0018261, 0.0035834, -0.0016580, 0.0038319, -0.0037676, 0.0029917
7: -0.0074370, -0.0000698, -0.0077754, -0.0002987, -0.0040745, 0.0051311
8: 0.9839751, 0.9891647, 0.9837368, 0.9890034, -0.0028702, 0.0036144
9: -0.0060517, -0.0013410, -0.0059054, -0.0011246, -0.0032810, 0.0026053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018152
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018152
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025791, 0.0049901, 0.0025996, 0.0049615, -0.0013024, 0.0014483
1: 0.0016949, 0.0020432, 0.0016979, 0.0020391, -0.0001882, 0.0002092
2: 0.0116009, 0.0129339, 0.0116168, 0.0129226, -0.0008007, 0.0007201
3: -0.0026822, -0.0013036, -0.0026659, -0.0013153, -0.0008282, 0.0007447
4: -0.0026258, -0.0011333, -0.0026131, -0.0011510, -0.0008062, 0.0008965
5: 0.0051858, 0.0065982, 0.0052026, 0.0065862, -0.0008484, 0.0007630
6: -0.0017244, 0.0038795, -0.0016580, 0.0038319, -0.0033663, 0.0030272
7: -0.0078402, -0.0002082, -0.0077754, -0.0002987, -0.0041228, 0.0045845
8: 0.9836910, 0.9890672, 0.9837368, 0.9890034, -0.0029042, 0.0032295
9: -0.0059632, -0.0010831, -0.0059054, -0.0011246, -0.0029315, 0.0026363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0017803
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0017803
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027149, 0.0050254, 0.0026785, 0.0050636, -0.0013326, 0.0015527
1: 0.0017145, 0.0020483, 0.0017093, 0.0020538, -0.0001925, 0.0002243
2: 0.0115814, 0.0128589, 0.0115603, 0.0128790, -0.0008584, 0.0007367
3: -0.0027024, -0.0013812, -0.0027242, -0.0013604, -0.0008878, 0.0007620
4: -0.0025417, -0.0011114, -0.0025642, -0.0010878, -0.0008249, 0.0009611
5: 0.0051652, 0.0065187, 0.0051428, 0.0065400, -0.0009096, 0.0007806
6: -0.0018065, 0.0035639, -0.0018952, 0.0036483, -0.0036089, 0.0030973
7: -0.0074105, -0.0000964, -0.0075254, 0.0000243, -0.0042182, 0.0049150
8: 0.9839938, 0.9891460, 0.9839128, 0.9892311, -0.0029714, 0.0034622
9: -0.0060347, -0.0013579, -0.0061119, -0.0012844, -0.0031428, 0.0026972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017885
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017885
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026735, 0.0050523, 0.0026785, 0.0050636, -0.0012322, 0.0014787
1: 0.0017085, 0.0020522, 0.0017093, 0.0020538, -0.0001780, 0.0002136
2: 0.0115666, 0.0128817, 0.0115603, 0.0128790, -0.0008176, 0.0006813
3: -0.0027178, -0.0013576, -0.0027242, -0.0013604, -0.0008456, 0.0007046
4: -0.0025673, -0.0010948, -0.0025642, -0.0010878, -0.0007628, 0.0009154
5: 0.0051494, 0.0065429, 0.0051428, 0.0065400, -0.0008662, 0.0007218
6: -0.0018690, 0.0036600, -0.0018952, 0.0036483, -0.0034370, 0.0028640
7: -0.0075413, -0.0000112, -0.0075254, 0.0000243, -0.0039005, 0.0046809
8: 0.9839016, 0.9892060, 0.9839128, 0.9892311, -0.0027476, 0.0032973
9: -0.0060892, -0.0012742, -0.0061119, -0.0012844, -0.0029931, 0.0024941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017723
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017723
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027077, 0.0050711, 0.0025578, 0.0050131, -0.0013678, 0.0016757
1: 0.0017135, 0.0020549, 0.0016918, 0.0020466, -0.0001976, 0.0002421
2: 0.0115562, 0.0128629, 0.0115882, 0.0129457, -0.0009265, 0.0007562
3: -0.0027285, -0.0013771, -0.0026954, -0.0012914, -0.0009582, 0.0007821
4: -0.0025462, -0.0010832, -0.0026390, -0.0011191, -0.0008467, 0.0010373
5: 0.0051384, 0.0065229, 0.0051724, 0.0066107, -0.0009816, 0.0008013
6: -0.0019126, 0.0035806, -0.0017779, 0.0039291, -0.0038948, 0.0031792
7: -0.0074332, 0.0000480, -0.0079077, -0.0001353, -0.0043298, 0.0053044
8: 0.9839778, 0.9892477, 0.9836435, 0.9891185, -0.0030500, 0.0037365
9: -0.0061271, -0.0013433, -0.0060098, -0.0010399, -0.0033918, 0.0027686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018498
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018498
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025873, 0.0050291, 0.0025578, 0.0050131, -0.0013780, 0.0015068
1: 0.0016961, 0.0020489, 0.0016918, 0.0020466, -0.0001991, 0.0002177
2: 0.0115794, 0.0129294, 0.0115882, 0.0129457, -0.0008331, 0.0007619
3: -0.0027045, -0.0013083, -0.0026954, -0.0012914, -0.0008616, 0.0007880
4: -0.0026207, -0.0011092, -0.0026390, -0.0011191, -0.0008530, 0.0009328
5: 0.0051630, 0.0065934, 0.0051724, 0.0066107, -0.0008827, 0.0008072
6: -0.0018151, 0.0038604, -0.0017779, 0.0039291, -0.0035023, 0.0032029
7: -0.0078143, -0.0000847, -0.0079077, -0.0001353, -0.0043621, 0.0047698
8: 0.9837093, 0.9891542, 0.9836435, 0.9891185, -0.0030728, 0.0033600
9: -0.0060422, -0.0010997, -0.0060098, -0.0010399, -0.0030500, 0.0027892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018123
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018122
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026952, 0.0049869, 0.0026821, 0.0050358, -0.0014161, 0.0015099
1: 0.0017117, 0.0020428, 0.0017098, 0.0020498, -0.0002046, 0.0002181
2: 0.0116027, 0.0128697, 0.0115757, 0.0128770, -0.0008348, 0.0007829
3: -0.0026804, -0.0013700, -0.0027083, -0.0013625, -0.0008634, 0.0008098
4: -0.0025539, -0.0011353, -0.0025620, -0.0011050, -0.0008766, 0.0009346
5: 0.0051877, 0.0065302, 0.0051591, 0.0065379, -0.0008845, 0.0008296
6: -0.0017169, 0.0036095, -0.0018307, 0.0036401, -0.0035094, 0.0032915
7: -0.0074726, -0.0002184, -0.0075142, -0.0000635, -0.0044827, 0.0047795
8: 0.9839500, 0.9890600, 0.9839208, 0.9891691, -0.0031577, 0.0033668
9: -0.0059567, -0.0013182, -0.0060557, -0.0012916, -0.0030561, 0.0028664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017759
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017759
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026581, 0.0050153, 0.0026821, 0.0050358, -0.0013208, 0.0014289
1: 0.0017063, 0.0020469, 0.0017098, 0.0020498, -0.0001908, 0.0002064
2: 0.0115870, 0.0128903, 0.0115757, 0.0128770, -0.0007900, 0.0007302
3: -0.0026966, -0.0013487, -0.0027083, -0.0013625, -0.0008171, 0.0007552
4: -0.0025769, -0.0011177, -0.0025620, -0.0011050, -0.0008176, 0.0008845
5: 0.0051711, 0.0065520, 0.0051591, 0.0065379, -0.0008371, 0.0007737
6: -0.0017830, 0.0036959, -0.0018307, 0.0036401, -0.0033212, 0.0030699
7: -0.0075902, -0.0001284, -0.0075142, -0.0000635, -0.0041809, 0.0045232
8: 0.9838672, 0.9891235, 0.9839208, 0.9891691, -0.0029451, 0.0031862
9: -0.0060143, -0.0012430, -0.0060557, -0.0012916, -0.0028923, 0.0026734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017600
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017600
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026887, 0.0050345, 0.0025590, 0.0049818, -0.0014465, 0.0016356
1: 0.0017107, 0.0020496, 0.0016920, 0.0020420, -0.0002090, 0.0002363
2: 0.0115764, 0.0128733, 0.0116055, 0.0129451, -0.0009043, 0.0007997
3: -0.0027076, -0.0013663, -0.0026775, -0.0012921, -0.0009353, 0.0008271
4: -0.0025579, -0.0011059, -0.0026382, -0.0011384, -0.0008954, 0.0010125
5: 0.0051599, 0.0065340, 0.0051907, 0.0066100, -0.0009581, 0.0008474
6: -0.0018275, 0.0036246, -0.0017052, 0.0039261, -0.0038016, 0.0033621
7: -0.0074931, -0.0000679, -0.0079038, -0.0002344, -0.0045789, 0.0051775
8: 0.9839355, 0.9891660, 0.9836462, 0.9890487, -0.0032255, 0.0036471
9: -0.0060529, -0.0013050, -0.0059464, -0.0010425, -0.0033106, 0.0029279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018284
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018284
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025661, 0.0049907, 0.0025590, 0.0049818, -0.0014716, 0.0014622
1: 0.0016930, 0.0020433, 0.0016920, 0.0020420, -0.0002126, 0.0002112
2: 0.0116006, 0.0129411, 0.0116055, 0.0129451, -0.0008084, 0.0008136
3: -0.0026825, -0.0012961, -0.0026775, -0.0012921, -0.0008361, 0.0008415
4: -0.0026338, -0.0011330, -0.0026382, -0.0011384, -0.0009110, 0.0009051
5: 0.0051855, 0.0066058, 0.0051907, 0.0066100, -0.0008565, 0.0008621
6: -0.0017258, 0.0039097, -0.0017052, 0.0039261, -0.0033985, 0.0034205
7: -0.0078814, -0.0002064, -0.0079038, -0.0002344, -0.0046584, 0.0046284
8: 0.9836620, 0.9890685, 0.9836462, 0.9890487, -0.0032815, 0.0032603
9: -0.0059644, -0.0010567, -0.0059464, -0.0010425, -0.0029595, 0.0029787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0017997
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0017997
time: 0.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.49 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017947, upper bound: 0.0018495
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017947, upper bound: 0.0018495
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018393, upper bound: 0.0017954
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018393, upper bound: 0.0018649
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017925, upper bound: 0.0018416
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017925, upper bound: 0.0018416
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018374, upper bound: 0.0017742
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018374, upper bound: 0.0018589
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017948, upper bound: 0.0017822
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017948, upper bound: 0.0018291
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018396, upper bound: 0.0017822
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018396, upper bound: 0.0018447
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017926, upper bound: 0.0017618
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017926, upper bound: 0.0018269
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018378, upper bound: 0.0017618
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018378, upper bound: 0.0018444
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017947, upper bound: 0.0018624
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017947, upper bound: 0.0018624
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018395, upper bound: 0.0018059
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018395, upper bound: 0.0018792
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017931, upper bound: 0.0018558
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017931, upper bound: 0.0018558
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018383, upper bound: 0.0017929
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018383, upper bound: 0.0018741
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017948, upper bound: 0.0017948
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017948, upper bound: 0.0018397
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018397, upper bound: 0.0017948
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018397, upper bound: 0.0018578
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0017807
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017933, upper bound: 0.0018376
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018384, upper bound: 0.0017807
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018384, upper bound: 0.0018563
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017706
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017706
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018332
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018332
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0017921
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0017921
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017581
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017581
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018152
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0018152
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0017803
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017799, upper bound: 0.0017803
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017885
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017885
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017723
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017723
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018498
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018498
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018123
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018122
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017759
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017759
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017600
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0018256, upper bound: 0.0017600
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018284
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0018284
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0017997
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 8, lower bound: -0.0017807, upper bound: 0.0017997

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026680, 0.0050002, 0.0026542, 0.0049498, -0.0012121, 0.0011615
1: 0.0017078, 0.0020447, 0.0017058, 0.0020374, -0.0001751, 0.0001678
2: 0.0115954, 0.0128848, 0.0116232, 0.0128924, -0.0006422, 0.0006701
3: -0.0026880, -0.0013544, -0.0026592, -0.0013465, -0.0006642, 0.0006931
4: -0.0025707, -0.0011271, -0.0025793, -0.0011583, -0.0007503, 0.0007190
5: 0.0051799, 0.0065461, 0.0052095, 0.0065542, -0.0006804, 0.0007100
6: -0.0017479, 0.0036727, -0.0016307, 0.0037050, -0.0026996, 0.0028171
7: -0.0075587, -0.0001763, -0.0076026, -0.0003358, -0.0038367, 0.0036767
8: 0.9838895, 0.9890897, 0.9838585, 0.9889773, -0.0027027, 0.0025899
9: -0.0059836, -0.0012631, -0.0058816, -0.0012351, -0.0023510, 0.0024533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026680, 0.0050002, 0.0026334, 0.0049106, -0.0012110, 0.0012456
1: 0.0017078, 0.0020447, 0.0017028, 0.0020317, -0.0001750, 0.0001800
2: 0.0115954, 0.0128848, 0.0116449, 0.0129039, -0.0006887, 0.0006696
3: -0.0026880, -0.0013544, -0.0026368, -0.0013346, -0.0007122, 0.0006925
4: -0.0025707, -0.0011271, -0.0025921, -0.0011825, -0.0007497, 0.0007710
5: 0.0051799, 0.0065461, 0.0052324, 0.0065664, -0.0007297, 0.0007094
6: -0.0017479, 0.0036727, -0.0015397, 0.0037532, -0.0028951, 0.0028148
7: -0.0075587, -0.0001763, -0.0076683, -0.0004598, -0.0038335, 0.0039429
8: 0.9838895, 0.9890897, 0.9838122, 0.9888900, -0.0027004, 0.0027774
9: -0.0059836, -0.0012631, -0.0058024, -0.0011931, -0.0025212, 0.0024513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025463, 0.0049528, 0.0027350, 0.0050367, -0.0014061, 0.0010798
1: 0.0016902, 0.0020378, 0.0017174, 0.0020500, -0.0002031, 0.0001560
2: 0.0116216, 0.0129521, 0.0115752, 0.0128478, -0.0005970, 0.0007774
3: -0.0026609, -0.0012848, -0.0027088, -0.0013927, -0.0006174, 0.0008040
4: -0.0026461, -0.0011564, -0.0025293, -0.0011045, -0.0008704, 0.0006684
5: 0.0052077, 0.0066174, 0.0051586, 0.0065069, -0.0006325, 0.0008237
6: -0.0016377, 0.0039557, -0.0018326, 0.0035171, -0.0025098, 0.0032682
7: -0.0079440, -0.0003262, -0.0073467, -0.0000608, -0.0044510, 0.0034181
8: 0.9836179, 0.9889840, 0.9840387, 0.9891710, -0.0031354, 0.0024078
9: -0.0058877, -0.0010167, -0.0060574, -0.0013987, -0.0021856, 0.0028461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025463, 0.0049528, 0.0026114, 0.0049882, -0.0012402, 0.0010643
1: 0.0016902, 0.0020378, 0.0016996, 0.0020429, -0.0001792, 0.0001538
2: 0.0116216, 0.0129521, 0.0116020, 0.0129161, -0.0005884, 0.0006857
3: -0.0026609, -0.0012848, -0.0026811, -0.0013220, -0.0006086, 0.0007092
4: -0.0026461, -0.0011564, -0.0026058, -0.0011345, -0.0007677, 0.0006588
5: 0.0052077, 0.0066174, 0.0051870, 0.0065793, -0.0006235, 0.0007265
6: -0.0016377, 0.0039557, -0.0017199, 0.0038044, -0.0024737, 0.0028825
7: -0.0079440, -0.0003262, -0.0077380, -0.0002144, -0.0039258, 0.0033690
8: 0.9836179, 0.9889840, 0.9837630, 0.9890628, -0.0027654, 0.0023732
9: -0.0058877, -0.0010167, -0.0059593, -0.0011485, -0.0021542, 0.0025102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018149
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018149
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026680, 0.0050002, 0.0027149, 0.0050254, -0.0014373, 0.0012604
1: 0.0017078, 0.0020447, 0.0017145, 0.0020483, -0.0002077, 0.0001821
2: 0.0115954, 0.0128848, 0.0115814, 0.0128589, -0.0006968, 0.0007947
3: -0.0026880, -0.0013544, -0.0027024, -0.0013812, -0.0007207, 0.0008219
4: -0.0025707, -0.0011271, -0.0025417, -0.0011114, -0.0008897, 0.0007802
5: 0.0051799, 0.0065461, 0.0051652, 0.0065187, -0.0007383, 0.0008420
6: -0.0017479, 0.0036727, -0.0018065, 0.0035639, -0.0029294, 0.0033407
7: -0.0075587, -0.0001763, -0.0074105, -0.0000964, -0.0045498, 0.0039896
8: 0.9838895, 0.9890897, 0.9839938, 0.9891460, -0.0032050, 0.0028104
9: -0.0059836, -0.0012631, -0.0060347, -0.0013579, -0.0025511, 0.0029093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026680, 0.0050002, 0.0026952, 0.0049869, -0.0014113, 0.0013174
1: 0.0017078, 0.0020447, 0.0017117, 0.0020428, -0.0002039, 0.0001903
2: 0.0115954, 0.0128848, 0.0116027, 0.0128697, -0.0007283, 0.0007803
3: -0.0026880, -0.0013544, -0.0026804, -0.0013700, -0.0007533, 0.0008070
4: -0.0025707, -0.0011271, -0.0025539, -0.0011353, -0.0008736, 0.0008155
5: 0.0051799, 0.0065461, 0.0051877, 0.0065302, -0.0007717, 0.0008268
6: -0.0017479, 0.0036727, -0.0017169, 0.0036095, -0.0030619, 0.0032803
7: -0.0075587, -0.0001763, -0.0074726, -0.0002184, -0.0044675, 0.0041701
8: 0.9838895, 0.9890897, 0.9839500, 0.9890600, -0.0031470, 0.0029375
9: -0.0059836, -0.0012631, -0.0059567, -0.0013182, -0.0026665, 0.0028567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025463, 0.0049528, 0.0027959, 0.0051052, -0.0016772, 0.0011833
1: 0.0016902, 0.0020378, 0.0017262, 0.0020599, -0.0002423, 0.0001710
2: 0.0116216, 0.0129521, 0.0115373, 0.0128141, -0.0006542, 0.0009273
3: -0.0026609, -0.0012848, -0.0027480, -0.0014275, -0.0006766, 0.0009591
4: -0.0026461, -0.0011564, -0.0024916, -0.0010620, -0.0010382, 0.0007325
5: 0.0052077, 0.0066174, 0.0051184, 0.0064712, -0.0006932, 0.0009825
6: -0.0016377, 0.0039557, -0.0019920, 0.0033755, -0.0027503, 0.0038983
7: -0.0079440, -0.0003262, -0.0071539, 0.0001562, -0.0053092, 0.0037456
8: 0.9836179, 0.9889840, 0.9841745, 0.9893240, -0.0037399, 0.0026385
9: -0.0058877, -0.0010167, -0.0061962, -0.0015220, -0.0023951, 0.0033948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025463, 0.0049528, 0.0026667, 0.0050627, -0.0015111, 0.0011722
1: 0.0016902, 0.0020378, 0.0017076, 0.0020537, -0.0002183, 0.0001694
2: 0.0116216, 0.0129521, 0.0115608, 0.0128855, -0.0006481, 0.0008355
3: -0.0026609, -0.0012848, -0.0027237, -0.0013537, -0.0006703, 0.0008641
4: -0.0026461, -0.0011564, -0.0025715, -0.0010884, -0.0009354, 0.0007256
5: 0.0052077, 0.0066174, 0.0051433, 0.0065469, -0.0006867, 0.0008852
6: -0.0016377, 0.0039557, -0.0018931, 0.0036757, -0.0027246, 0.0035122
7: -0.0079440, -0.0003262, -0.0075627, 0.0000215, -0.0047834, 0.0037107
8: 0.9836179, 0.9889840, 0.9838865, 0.9892291, -0.0033695, 0.0026139
9: -0.0058877, -0.0010167, -0.0061101, -0.0012605, -0.0023727, 0.0030586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017968
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017968
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026468, 0.0049649, 0.0027361, 0.0050085, -0.0013184, 0.0010666
1: 0.0017047, 0.0020396, 0.0017176, 0.0020459, -0.0001905, 0.0001541
2: 0.0116149, 0.0128965, 0.0115908, 0.0128471, -0.0005897, 0.0007289
3: -0.0026678, -0.0013423, -0.0026927, -0.0013934, -0.0006099, 0.0007539
4: -0.0025839, -0.0011489, -0.0025286, -0.0011219, -0.0008161, 0.0006602
5: 0.0052006, 0.0065586, 0.0051751, 0.0065062, -0.0006248, 0.0007723
6: -0.0016657, 0.0037221, -0.0017671, 0.0035144, -0.0024790, 0.0030643
7: -0.0076260, -0.0002881, -0.0073431, -0.0001500, -0.0041733, 0.0033762
8: 0.9838420, 0.9890109, 0.9840412, 0.9891082, -0.0029397, 0.0023783
9: -0.0059121, -0.0012201, -0.0060004, -0.0014010, -0.0021589, 0.0026685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0017872
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0017872
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026468, 0.0049649, 0.0026126, 0.0049584, -0.0013122, 0.0012431
1: 0.0017047, 0.0020396, 0.0016997, 0.0020387, -0.0001896, 0.0001796
2: 0.0116149, 0.0128965, 0.0116185, 0.0129154, -0.0006873, 0.0007255
3: -0.0026678, -0.0013423, -0.0026641, -0.0013227, -0.0007108, 0.0007503
4: -0.0025839, -0.0011489, -0.0026051, -0.0011529, -0.0008122, 0.0007695
5: 0.0052006, 0.0065586, 0.0052044, 0.0065786, -0.0007282, 0.0007687
6: -0.0016657, 0.0037221, -0.0016508, 0.0038017, -0.0028893, 0.0030498
7: -0.0076260, -0.0002881, -0.0077342, -0.0003085, -0.0041536, 0.0039350
8: 0.9838420, 0.9890109, 0.9837657, 0.9889965, -0.0029259, 0.0027719
9: -0.0059121, -0.0012201, -0.0058991, -0.0011509, -0.0025161, 0.0026559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018291
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018291
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025258, 0.0049137, 0.0027361, 0.0050085, -0.0014643, 0.0010705
1: 0.0016872, 0.0020322, 0.0017176, 0.0020459, -0.0002116, 0.0001547
2: 0.0116432, 0.0129634, 0.0115908, 0.0128471, -0.0005918, 0.0008096
3: -0.0026385, -0.0012731, -0.0026927, -0.0013934, -0.0006121, 0.0008373
4: -0.0026588, -0.0011806, -0.0025286, -0.0011219, -0.0009065, 0.0006627
5: 0.0052306, 0.0066294, 0.0051751, 0.0065062, -0.0006271, 0.0008578
6: -0.0015469, 0.0040033, -0.0017671, 0.0035144, -0.0024881, 0.0034035
7: -0.0080089, -0.0004500, -0.0073431, -0.0001500, -0.0046353, 0.0033886
8: 0.9835722, 0.9888968, 0.9840412, 0.9891082, -0.0032652, 0.0023870
9: -0.0058086, -0.0009753, -0.0060004, -0.0014010, -0.0021668, 0.0029640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025258, 0.0049137, 0.0026126, 0.0049584, -0.0013124, 0.0010605
1: 0.0016872, 0.0020322, 0.0016997, 0.0020387, -0.0001896, 0.0001532
2: 0.0116432, 0.0129634, 0.0116185, 0.0129154, -0.0005863, 0.0007256
3: -0.0026385, -0.0012731, -0.0026641, -0.0013227, -0.0006064, 0.0007505
4: -0.0026588, -0.0011806, -0.0026051, -0.0011529, -0.0008124, 0.0006565
5: 0.0052306, 0.0066294, 0.0052044, 0.0065786, -0.0006212, 0.0007688
6: -0.0015469, 0.0040033, -0.0016508, 0.0038017, -0.0024648, 0.0030504
7: -0.0080089, -0.0004500, -0.0077342, -0.0003085, -0.0041544, 0.0033569
8: 0.9835722, 0.9888968, 0.9837657, 0.9889965, -0.0029265, 0.0023647
9: -0.0058086, -0.0009753, -0.0058991, -0.0011509, -0.0021465, 0.0026565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018023
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018023
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026468, 0.0049649, 0.0027971, 0.0050781, -0.0015730, 0.0011699
1: 0.0017047, 0.0020396, 0.0017264, 0.0020559, -0.0002273, 0.0001690
2: 0.0116149, 0.0128965, 0.0115523, 0.0128134, -0.0006468, 0.0008697
3: -0.0026678, -0.0013423, -0.0027325, -0.0014282, -0.0006690, 0.0008995
4: -0.0025839, -0.0011489, -0.0024908, -0.0010788, -0.0009737, 0.0007242
5: 0.0052006, 0.0065586, 0.0051343, 0.0064705, -0.0006853, 0.0009215
6: -0.0016657, 0.0037221, -0.0019289, 0.0033727, -0.0027193, 0.0036561
7: -0.0076260, -0.0002881, -0.0071500, 0.0000703, -0.0049793, 0.0037034
8: 0.9838420, 0.9890109, 0.9841773, 0.9892634, -0.0035075, 0.0026087
9: -0.0059121, -0.0012201, -0.0061413, -0.0015244, -0.0023680, 0.0031839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017675
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017675
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026468, 0.0049649, 0.0026683, 0.0050340, -0.0015253, 0.0013249
1: 0.0017047, 0.0020396, 0.0017078, 0.0020496, -0.0002204, 0.0001914
2: 0.0116149, 0.0128965, 0.0115767, 0.0128846, -0.0007325, 0.0008433
3: -0.0026678, -0.0013423, -0.0027073, -0.0013545, -0.0007576, 0.0008722
4: -0.0025839, -0.0011489, -0.0025706, -0.0011061, -0.0009442, 0.0008202
5: 0.0052006, 0.0065586, 0.0051601, 0.0065460, -0.0007761, 0.0008935
6: -0.0016657, 0.0037221, -0.0018265, 0.0036722, -0.0030795, 0.0035453
7: -0.0076260, -0.0002881, -0.0075580, -0.0000692, -0.0048284, 0.0041940
8: 0.9838420, 0.9890109, 0.9838899, 0.9891652, -0.0034012, 0.0029544
9: -0.0059121, -0.0012201, -0.0060521, -0.0012636, -0.0026818, 0.0030874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025258, 0.0049137, 0.0027971, 0.0050781, -0.0017190, 0.0011738
1: 0.0016872, 0.0020322, 0.0017264, 0.0020559, -0.0002483, 0.0001696
2: 0.0116432, 0.0129634, 0.0115523, 0.0128134, -0.0006490, 0.0009504
3: -0.0026385, -0.0012731, -0.0027325, -0.0014282, -0.0006712, 0.0009829
4: -0.0026588, -0.0011806, -0.0024908, -0.0010788, -0.0010641, 0.0007266
5: 0.0052306, 0.0066294, 0.0051343, 0.0064705, -0.0006876, 0.0010070
6: -0.0015469, 0.0040033, -0.0019289, 0.0033727, -0.0027283, 0.0039953
7: -0.0080089, -0.0004500, -0.0071500, 0.0000703, -0.0054413, 0.0037157
8: 0.9835722, 0.9888968, 0.9841773, 0.9892634, -0.0038330, 0.0026174
9: -0.0058086, -0.0009753, -0.0061413, -0.0015244, -0.0023759, 0.0034793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017618
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017617
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025258, 0.0049137, 0.0026680, 0.0050340, -0.0015673, 0.0011683
1: 0.0016872, 0.0020322, 0.0017077, 0.0020496, -0.0002264, 0.0001688
2: 0.0116432, 0.0129634, 0.0115767, 0.0128848, -0.0006459, 0.0008665
3: -0.0026385, -0.0012731, -0.0027073, -0.0013544, -0.0006680, 0.0008962
4: -0.0026588, -0.0011806, -0.0025708, -0.0011061, -0.0009702, 0.0007232
5: 0.0052306, 0.0066294, 0.0051601, 0.0065462, -0.0006844, 0.0009181
6: -0.0015469, 0.0040033, -0.0018265, 0.0036729, -0.0027154, 0.0036429
7: -0.0080089, -0.0004500, -0.0075589, -0.0000692, -0.0049613, 0.0036982
8: 0.9835722, 0.9888968, 0.9838892, 0.9891652, -0.0034948, 0.0026051
9: -0.0058086, -0.0009753, -0.0060521, -0.0012630, -0.0023647, 0.0031724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017849
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017848
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026539, 0.0050006, 0.0026208, 0.0049726, -0.0013620, 0.0011767
1: 0.0017057, 0.0020447, 0.0017009, 0.0020407, -0.0001968, 0.0001700
2: 0.0115952, 0.0128926, 0.0116107, 0.0129109, -0.0006506, 0.0007530
3: -0.0026882, -0.0013463, -0.0026722, -0.0013274, -0.0006729, 0.0007788
4: -0.0025795, -0.0011269, -0.0025999, -0.0011442, -0.0008431, 0.0007284
5: 0.0051797, 0.0065544, 0.0051961, 0.0065738, -0.0006893, 0.0007979
6: -0.0017487, 0.0037056, -0.0016836, 0.0037825, -0.0027350, 0.0031658
7: -0.0076034, -0.0001752, -0.0077081, -0.0002638, -0.0043115, 0.0037249
8: 0.9838579, 0.9890904, 0.9837841, 0.9890280, -0.0030371, 0.0026239
9: -0.0059843, -0.0012345, -0.0059277, -0.0011676, -0.0023818, 0.0027569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018621
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018493
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026539, 0.0050006, 0.0026017, 0.0049302, -0.0013624, 0.0012602
1: 0.0017057, 0.0020447, 0.0016982, 0.0020346, -0.0001968, 0.0001821
2: 0.0115952, 0.0128926, 0.0116341, 0.0129214, -0.0006967, 0.0007533
3: -0.0026882, -0.0013463, -0.0026480, -0.0013165, -0.0007206, 0.0007790
4: -0.0025795, -0.0011269, -0.0026118, -0.0011704, -0.0008434, 0.0007801
5: 0.0051797, 0.0065544, 0.0052209, 0.0065850, -0.0007382, 0.0007981
6: -0.0017487, 0.0037056, -0.0015853, 0.0038269, -0.0029290, 0.0031667
7: -0.0076034, -0.0001752, -0.0077686, -0.0003977, -0.0043127, 0.0039891
8: 0.9838579, 0.9890904, 0.9837415, 0.9889337, -0.0030380, 0.0028100
9: -0.0059843, -0.0012345, -0.0058420, -0.0011289, -0.0025507, 0.0027577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018621
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018493
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025314, 0.0049533, 0.0027010, 0.0050603, -0.0015742, 0.0010961
1: 0.0016880, 0.0020379, 0.0017125, 0.0020534, -0.0002274, 0.0001584
2: 0.0116213, 0.0129603, 0.0115622, 0.0128666, -0.0006060, 0.0008703
3: -0.0026611, -0.0012763, -0.0027223, -0.0013733, -0.0006268, 0.0009001
4: -0.0026553, -0.0011561, -0.0025503, -0.0010899, -0.0009744, 0.0006785
5: 0.0052074, 0.0066262, 0.0051447, 0.0065268, -0.0006421, 0.0009222
6: -0.0016388, 0.0039904, -0.0018875, 0.0035961, -0.0025477, 0.0036588
7: -0.0079913, -0.0003249, -0.0074543, 0.0000140, -0.0049830, 0.0034697
8: 0.9835846, 0.9889851, 0.9839629, 0.9892237, -0.0035101, 0.0024441
9: -0.0058886, -0.0009865, -0.0061053, -0.0013298, -0.0022186, 0.0031863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018059
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017934
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025314, 0.0049533, 0.0025763, 0.0050099, -0.0014130, 0.0010800
1: 0.0016880, 0.0020379, 0.0016945, 0.0020461, -0.0002041, 0.0001560
2: 0.0116213, 0.0129603, 0.0115900, 0.0129355, -0.0005971, 0.0007812
3: -0.0026611, -0.0012763, -0.0026935, -0.0013020, -0.0006176, 0.0008080
4: -0.0026553, -0.0011561, -0.0026275, -0.0011211, -0.0008747, 0.0006685
5: 0.0052074, 0.0066262, 0.0051743, 0.0065998, -0.0006327, 0.0008277
6: -0.0016388, 0.0039904, -0.0017704, 0.0038858, -0.0025102, 0.0032842
7: -0.0079913, -0.0003249, -0.0078489, -0.0001455, -0.0044728, 0.0034187
8: 0.9835846, 0.9889851, 0.9836849, 0.9891114, -0.0031507, 0.0024082
9: -0.0058886, -0.0009865, -0.0060033, -0.0010775, -0.0021860, 0.0028600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018264
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018154
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026539, 0.0050006, 0.0026735, 0.0050523, -0.0015839, 0.0012745
1: 0.0017057, 0.0020447, 0.0017085, 0.0020522, -0.0002288, 0.0001841
2: 0.0115952, 0.0128926, 0.0115666, 0.0128817, -0.0007046, 0.0008757
3: -0.0026882, -0.0013463, -0.0027178, -0.0013576, -0.0007288, 0.0009057
4: -0.0025795, -0.0011269, -0.0025673, -0.0010948, -0.0009804, 0.0007889
5: 0.0051797, 0.0065544, 0.0051494, 0.0065429, -0.0007466, 0.0009278
6: -0.0017487, 0.0037056, -0.0018690, 0.0036600, -0.0029623, 0.0036814
7: -0.0076034, -0.0001752, -0.0075413, -0.0000112, -0.0050137, 0.0040344
8: 0.9838579, 0.9890904, 0.9839016, 0.9892060, -0.0035318, 0.0028419
9: -0.0059843, -0.0012345, -0.0060892, -0.0012742, -0.0025797, 0.0032059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018545
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018445
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026539, 0.0050006, 0.0026581, 0.0050153, -0.0015615, 0.0013311
1: 0.0017057, 0.0020447, 0.0017063, 0.0020469, -0.0002256, 0.0001923
2: 0.0115952, 0.0128926, 0.0115870, 0.0128903, -0.0007359, 0.0008633
3: -0.0026882, -0.0013463, -0.0026966, -0.0013487, -0.0007611, 0.0008929
4: -0.0025795, -0.0011269, -0.0025769, -0.0011177, -0.0009666, 0.0008240
5: 0.0051797, 0.0065544, 0.0051711, 0.0065520, -0.0007798, 0.0009147
6: -0.0017487, 0.0037056, -0.0017830, 0.0036959, -0.0030939, 0.0036293
7: -0.0076034, -0.0001752, -0.0075902, -0.0001284, -0.0049428, 0.0042136
8: 0.9838579, 0.9890904, 0.9838672, 0.9891235, -0.0034818, 0.0029681
9: -0.0059843, -0.0012345, -0.0060143, -0.0012430, -0.0026943, 0.0031605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018545
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018445
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025314, 0.0049533, 0.0027530, 0.0051409, -0.0018195, 0.0011976
1: 0.0016880, 0.0020379, 0.0017200, 0.0020650, -0.0002629, 0.0001730
2: 0.0116213, 0.0129603, 0.0115176, 0.0128378, -0.0006621, 0.0010060
3: -0.0026611, -0.0012763, -0.0027684, -0.0014030, -0.0006848, 0.0010404
4: -0.0026553, -0.0011561, -0.0025181, -0.0010400, -0.0011263, 0.0007414
5: 0.0052074, 0.0066262, 0.0050975, 0.0064963, -0.0007016, 0.0010659
6: -0.0016388, 0.0039904, -0.0020748, 0.0034752, -0.0027837, 0.0042291
7: -0.0079913, -0.0003249, -0.0072896, 0.0002690, -0.0057596, 0.0037911
8: 0.9835846, 0.9889851, 0.9840789, 0.9894034, -0.0040572, 0.0026705
9: -0.0058886, -0.0009865, -0.0062684, -0.0014352, -0.0024241, 0.0036829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017919
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017766
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025314, 0.0049533, 0.0026316, 0.0050911, -0.0016567, 0.0011866
1: 0.0016880, 0.0020379, 0.0017025, 0.0020578, -0.0002394, 0.0001714
2: 0.0116213, 0.0129603, 0.0115451, 0.0129049, -0.0006561, 0.0009160
3: -0.0026611, -0.0012763, -0.0027400, -0.0013336, -0.0006785, 0.0009473
4: -0.0026553, -0.0011561, -0.0025933, -0.0010708, -0.0010255, 0.0007346
5: 0.0052074, 0.0066262, 0.0051267, 0.0065675, -0.0006951, 0.0009705
6: -0.0016388, 0.0039904, -0.0019592, 0.0037575, -0.0027581, 0.0038507
7: -0.0079913, -0.0003249, -0.0076741, 0.0001116, -0.0052443, 0.0037563
8: 0.9835846, 0.9889851, 0.9838080, 0.9892925, -0.0036942, 0.0026460
9: -0.0058886, -0.0009865, -0.0061677, -0.0011893, -0.0024019, 0.0033534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0018146
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017999
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026350, 0.0049653, 0.0027022, 0.0050324, -0.0014865, 0.0010826
1: 0.0017030, 0.0020396, 0.0017127, 0.0020493, -0.0002148, 0.0001564
2: 0.0116147, 0.0129030, 0.0115776, 0.0128659, -0.0005985, 0.0008218
3: -0.0026680, -0.0013355, -0.0027064, -0.0013739, -0.0006190, 0.0008500
4: -0.0025912, -0.0011487, -0.0025496, -0.0011071, -0.0009202, 0.0006701
5: 0.0052004, 0.0065655, 0.0051611, 0.0065261, -0.0006342, 0.0008708
6: -0.0016667, 0.0037495, -0.0018227, 0.0035934, -0.0025162, 0.0034550
7: -0.0076632, -0.0002868, -0.0074506, -0.0000743, -0.0047054, 0.0034269
8: 0.9838157, 0.9890119, 0.9839656, 0.9891616, -0.0033146, 0.0024140
9: -0.0059130, -0.0011963, -0.0060488, -0.0013323, -0.0021913, 0.0030088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0017979
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0017858
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026350, 0.0049653, 0.0025776, 0.0049785, -0.0014685, 0.0012586
1: 0.0017030, 0.0020396, 0.0016947, 0.0020416, -0.0002122, 0.0001818
2: 0.0116147, 0.0129030, 0.0116074, 0.0129348, -0.0006958, 0.0008119
3: -0.0026680, -0.0013355, -0.0026756, -0.0013027, -0.0007197, 0.0008397
4: -0.0025912, -0.0011487, -0.0026267, -0.0011405, -0.0009090, 0.0007791
5: 0.0052004, 0.0065655, 0.0051926, 0.0065991, -0.0007373, 0.0008602
6: -0.0016667, 0.0037495, -0.0016974, 0.0038830, -0.0029253, 0.0034132
7: -0.0076632, -0.0002868, -0.0078450, -0.0002449, -0.0046485, 0.0039840
8: 0.9838157, 0.9890119, 0.9836877, 0.9890413, -0.0032745, 0.0028064
9: -0.0059130, -0.0011963, -0.0059397, -0.0010801, -0.0025475, 0.0029724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018395
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018288
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025144, 0.0049142, 0.0027022, 0.0050324, -0.0016335, 0.0010867
1: 0.0016856, 0.0020323, 0.0017127, 0.0020493, -0.0002360, 0.0001570
2: 0.0116429, 0.0129697, 0.0115776, 0.0128659, -0.0006008, 0.0009031
3: -0.0026388, -0.0012666, -0.0027064, -0.0013739, -0.0006214, 0.0009341
4: -0.0026658, -0.0011803, -0.0025496, -0.0011071, -0.0010112, 0.0006727
5: 0.0052303, 0.0066361, 0.0051611, 0.0065261, -0.0006366, 0.0009569
6: -0.0015481, 0.0040297, -0.0018227, 0.0035934, -0.0025257, 0.0037967
7: -0.0080449, -0.0004484, -0.0074506, -0.0000743, -0.0051708, 0.0034398
8: 0.9835469, 0.9888980, 0.9839656, 0.9891616, -0.0036424, 0.0024231
9: -0.0058096, -0.0009522, -0.0060488, -0.0013323, -0.0021995, 0.0033063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017948
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017808
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025144, 0.0049142, 0.0025776, 0.0049785, -0.0014868, 0.0010757
1: 0.0016856, 0.0020323, 0.0016947, 0.0020416, -0.0002148, 0.0001554
2: 0.0116429, 0.0129697, 0.0116074, 0.0129348, -0.0005947, 0.0008220
3: -0.0026388, -0.0012666, -0.0026756, -0.0013027, -0.0006151, 0.0008502
4: -0.0026658, -0.0011803, -0.0026267, -0.0011405, -0.0009204, 0.0006659
5: 0.0052303, 0.0066361, 0.0051926, 0.0065991, -0.0006301, 0.0008710
6: -0.0015481, 0.0040297, -0.0016974, 0.0038830, -0.0025002, 0.0034557
7: -0.0080449, -0.0004484, -0.0078450, -0.0002449, -0.0047064, 0.0034050
8: 0.9835469, 0.9888980, 0.9836877, 0.9890413, -0.0033153, 0.0023986
9: -0.0058096, -0.0009522, -0.0059397, -0.0010801, -0.0021773, 0.0030094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018156
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018035
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026350, 0.0049653, 0.0027543, 0.0051142, -0.0017174, 0.0011840
1: 0.0017030, 0.0020396, 0.0017202, 0.0020611, -0.0002481, 0.0001711
2: 0.0116147, 0.0129030, 0.0115324, 0.0128371, -0.0006546, 0.0009495
3: -0.0026680, -0.0013355, -0.0027531, -0.0014037, -0.0006770, 0.0009820
4: -0.0025912, -0.0011487, -0.0025173, -0.0010565, -0.0010631, 0.0007329
5: 0.0052004, 0.0065655, 0.0051132, 0.0064956, -0.0006936, 0.0010061
6: -0.0016667, 0.0037495, -0.0020127, 0.0034723, -0.0027520, 0.0039917
7: -0.0076632, -0.0002868, -0.0072857, 0.0001844, -0.0054364, 0.0037480
8: 0.9838157, 0.9890119, 0.9840816, 0.9893438, -0.0038295, 0.0026402
9: -0.0059130, -0.0011963, -0.0062143, -0.0014377, -0.0023966, 0.0034762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017854
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017700
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026350, 0.0049653, 0.0026339, 0.0050628, -0.0016775, 0.0013392
1: 0.0017030, 0.0020396, 0.0017028, 0.0020537, -0.0002424, 0.0001935
2: 0.0116147, 0.0129030, 0.0115608, 0.0129037, -0.0007404, 0.0009275
3: -0.0026680, -0.0013355, -0.0027237, -0.0013349, -0.0007658, 0.0009592
4: -0.0025912, -0.0011487, -0.0025919, -0.0010883, -0.0010384, 0.0008290
5: 0.0052004, 0.0065655, 0.0051433, 0.0065661, -0.0007845, 0.0009827
6: -0.0016667, 0.0037495, -0.0018932, 0.0037521, -0.0031128, 0.0038990
7: -0.0076632, -0.0002868, -0.0076667, 0.0000217, -0.0053102, 0.0042393
8: 0.9838157, 0.9890119, 0.9838133, 0.9892291, -0.0037406, 0.0029863
9: -0.0059130, -0.0011963, -0.0061102, -0.0011940, -0.0027107, 0.0033955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018362
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018286
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025144, 0.0049142, 0.0027543, 0.0051142, -0.0018644, 0.0011881
1: 0.0016856, 0.0020323, 0.0017202, 0.0020611, -0.0002694, 0.0001716
2: 0.0116429, 0.0129697, 0.0115324, 0.0128371, -0.0006569, 0.0010308
3: -0.0026388, -0.0012666, -0.0027531, -0.0014037, -0.0006794, 0.0010661
4: -0.0026658, -0.0011803, -0.0025173, -0.0010565, -0.0011541, 0.0007355
5: 0.0052303, 0.0066361, 0.0051132, 0.0064956, -0.0006960, 0.0010922
6: -0.0015481, 0.0040297, -0.0020127, 0.0034723, -0.0027615, 0.0043334
7: -0.0080449, -0.0004484, -0.0072857, 0.0001844, -0.0059017, 0.0037609
8: 0.9835469, 0.9888980, 0.9840816, 0.9893438, -0.0041573, 0.0026493
9: -0.0058096, -0.0009522, -0.0062143, -0.0014377, -0.0024048, 0.0037737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017799
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017639
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025144, 0.0049142, 0.0026328, 0.0050628, -0.0017151, 0.0011822
1: 0.0016856, 0.0020323, 0.0017027, 0.0020537, -0.0002478, 0.0001708
2: 0.0116429, 0.0129697, 0.0115608, 0.0129042, -0.0006536, 0.0009482
3: -0.0026388, -0.0012666, -0.0027237, -0.0013343, -0.0006760, 0.0009807
4: -0.0026658, -0.0011803, -0.0025925, -0.0010883, -0.0010617, 0.0007318
5: 0.0052303, 0.0066361, 0.0051433, 0.0065667, -0.0006925, 0.0010047
6: -0.0015481, 0.0040297, -0.0018932, 0.0037546, -0.0027478, 0.0039864
7: -0.0080449, -0.0004484, -0.0076701, 0.0000217, -0.0054291, 0.0037423
8: 0.9835469, 0.9888980, 0.9838109, 0.9892291, -0.0038244, 0.0026361
9: -0.0058096, -0.0009522, -0.0061102, -0.0011919, -0.0023929, 0.0034715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0018030
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017877
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027149, 0.0050254, 0.0027329, 0.0050367, -0.0013059, 0.0013574
1: 0.0017145, 0.0020483, 0.0017171, 0.0020500, -0.0001887, 0.0001961
2: 0.0115814, 0.0128589, 0.0115752, 0.0128489, -0.0007505, 0.0007220
3: -0.0027024, -0.0013812, -0.0027088, -0.0013915, -0.0007762, 0.0007467
4: -0.0025417, -0.0011114, -0.0025306, -0.0011045, -0.0008084, 0.0008403
5: 0.0051652, 0.0065187, 0.0051586, 0.0065081, -0.0007952, 0.0007650
6: -0.0018065, 0.0035639, -0.0018326, 0.0035220, -0.0031550, 0.0030352
7: -0.0074105, -0.0000964, -0.0073534, -0.0000608, -0.0041337, 0.0042968
8: 0.9839938, 0.9891460, 0.9840339, 0.9891710, -0.0029118, 0.0030268
9: -0.0060347, -0.0013579, -0.0060574, -0.0013944, -0.0027475, 0.0026432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027149, 0.0050254, 0.0027939, 0.0051052, -0.0012429, 0.0011626
1: 0.0017145, 0.0020483, 0.0017259, 0.0020599, -0.0001796, 0.0001680
2: 0.0115814, 0.0128589, 0.0115373, 0.0128152, -0.0006428, 0.0006872
3: -0.0027024, -0.0013812, -0.0027480, -0.0014264, -0.0006648, 0.0007107
4: -0.0025417, -0.0011114, -0.0024928, -0.0010620, -0.0007694, 0.0007197
5: 0.0051652, 0.0065187, 0.0051184, 0.0064724, -0.0006810, 0.0007281
6: -0.0018065, 0.0035639, -0.0019920, 0.0033802, -0.0027022, 0.0028889
7: -0.0074105, -0.0000964, -0.0071603, 0.0001562, -0.0039344, 0.0036802
8: 0.9839938, 0.9891460, 0.9841700, 0.9893240, -0.0027715, 0.0025924
9: -0.0060347, -0.0013579, -0.0061962, -0.0015179, -0.0023532, 0.0025158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026735, 0.0050523, 0.0027329, 0.0050367, -0.0013991, 0.0014546
1: 0.0017085, 0.0020522, 0.0017171, 0.0020500, -0.0002021, 0.0002101
2: 0.0115666, 0.0128817, 0.0115752, 0.0128489, -0.0008042, 0.0007735
3: -0.0027178, -0.0013576, -0.0027088, -0.0013915, -0.0008318, 0.0008000
4: -0.0025673, -0.0010948, -0.0025306, -0.0011045, -0.0008661, 0.0009004
5: 0.0051494, 0.0065429, 0.0051586, 0.0065081, -0.0008521, 0.0008196
6: -0.0018690, 0.0036600, -0.0018326, 0.0035220, -0.0033809, 0.0032518
7: -0.0075413, -0.0000112, -0.0073534, -0.0000608, -0.0044287, 0.0046045
8: 0.9839016, 0.9892060, 0.9840339, 0.9891710, -0.0031197, 0.0032435
9: -0.0060892, -0.0012742, -0.0060574, -0.0013944, -0.0029442, 0.0028318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026735, 0.0050523, 0.0027939, 0.0051052, -0.0013608, 0.0012669
1: 0.0017085, 0.0020522, 0.0017259, 0.0020599, -0.0001966, 0.0001830
2: 0.0115666, 0.0128817, 0.0115373, 0.0128152, -0.0007004, 0.0007524
3: -0.0027178, -0.0013576, -0.0027480, -0.0014264, -0.0007244, 0.0007781
4: -0.0025673, -0.0010948, -0.0024928, -0.0010620, -0.0008424, 0.0007842
5: 0.0051494, 0.0065429, 0.0051184, 0.0064724, -0.0007421, 0.0007972
6: -0.0018690, 0.0036600, -0.0019920, 0.0033802, -0.0029445, 0.0031629
7: -0.0075413, -0.0000112, -0.0071603, 0.0001562, -0.0043076, 0.0040102
8: 0.9839016, 0.9892060, 0.9841700, 0.9893240, -0.0030344, 0.0028249
9: -0.0060892, -0.0012742, -0.0061962, -0.0015179, -0.0025642, 0.0027544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027228, 0.0050705, 0.0026114, 0.0049882, -0.0013333, 0.0015277
1: 0.0017157, 0.0020548, 0.0016996, 0.0020429, -0.0001926, 0.0002207
2: 0.0115565, 0.0128545, 0.0116020, 0.0129161, -0.0008446, 0.0007371
3: -0.0027281, -0.0013858, -0.0026811, -0.0013220, -0.0008735, 0.0007624
4: -0.0025368, -0.0010836, -0.0026058, -0.0011345, -0.0008253, 0.0009456
5: 0.0051388, 0.0065140, 0.0051870, 0.0065793, -0.0008949, 0.0007810
6: -0.0019112, 0.0035454, -0.0017199, 0.0038044, -0.0035507, 0.0030990
7: -0.0073852, 0.0000461, -0.0077380, -0.0002144, -0.0042205, 0.0048358
8: 0.9840117, 0.9892463, 0.9837630, 0.9890628, -0.0029730, 0.0034064
9: -0.0061258, -0.0013740, -0.0059593, -0.0011485, -0.0030921, 0.0026987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027228, 0.0050705, 0.0026670, 0.0050627, -0.0012795, 0.0012864
1: 0.0017157, 0.0020548, 0.0017076, 0.0020537, -0.0001849, 0.0001858
2: 0.0115565, 0.0128545, 0.0115608, 0.0128853, -0.0007112, 0.0007074
3: -0.0027281, -0.0013858, -0.0027237, -0.0013539, -0.0007356, 0.0007317
4: -0.0025368, -0.0010836, -0.0025713, -0.0010884, -0.0007921, 0.0007963
5: 0.0051388, 0.0065140, 0.0051433, 0.0065467, -0.0007536, 0.0007495
6: -0.0019112, 0.0035454, -0.0018931, 0.0036750, -0.0029900, 0.0029740
7: -0.0073852, 0.0000461, -0.0075618, 0.0000215, -0.0040503, 0.0040721
8: 0.9840117, 0.9892463, 0.9838872, 0.9892291, -0.0028531, 0.0028685
9: -0.0061258, -0.0013740, -0.0061101, -0.0012611, -0.0026038, 0.0025899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026015, 0.0050285, 0.0026114, 0.0049882, -0.0013244, 0.0013460
1: 0.0016981, 0.0020488, 0.0016996, 0.0020429, -0.0001913, 0.0001945
2: 0.0115797, 0.0129216, 0.0116020, 0.0129161, -0.0007442, 0.0007322
3: -0.0027042, -0.0013164, -0.0026811, -0.0013220, -0.0007696, 0.0007573
4: -0.0026119, -0.0011095, -0.0026058, -0.0011345, -0.0008198, 0.0008332
5: 0.0051633, 0.0065851, 0.0051870, 0.0065793, -0.0007885, 0.0007758
6: -0.0018137, 0.0038274, -0.0017199, 0.0038044, -0.0031284, 0.0030782
7: -0.0077693, -0.0000866, -0.0077380, -0.0002144, -0.0041923, 0.0042606
8: 0.9837410, 0.9891529, 0.9837630, 0.9890628, -0.0029531, 0.0030013
9: -0.0060410, -0.0011285, -0.0059593, -0.0011485, -0.0027244, 0.0026807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026015, 0.0050285, 0.0026670, 0.0050627, -0.0012781, 0.0011058
1: 0.0016981, 0.0020488, 0.0017076, 0.0020537, -0.0001847, 0.0001598
2: 0.0115797, 0.0129216, 0.0115608, 0.0128853, -0.0006113, 0.0007066
3: -0.0027042, -0.0013164, -0.0027237, -0.0013539, -0.0006323, 0.0007308
4: -0.0026119, -0.0011095, -0.0025713, -0.0010884, -0.0007912, 0.0006845
5: 0.0051633, 0.0065851, 0.0051433, 0.0065467, -0.0006478, 0.0007487
6: -0.0018137, 0.0038274, -0.0018931, 0.0036750, -0.0025701, 0.0029707
7: -0.0077693, -0.0000866, -0.0075618, 0.0000215, -0.0040458, 0.0035003
8: 0.9837410, 0.9891529, 0.9838872, 0.9892291, -0.0028500, 0.0024657
9: -0.0060410, -0.0011285, -0.0061101, -0.0012611, -0.0022382, 0.0025870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026952, 0.0049869, 0.0027361, 0.0050085, -0.0013716, 0.0012929
1: 0.0017117, 0.0020428, 0.0017176, 0.0020459, -0.0001982, 0.0001868
2: 0.0116027, 0.0128697, 0.0115908, 0.0128471, -0.0007148, 0.0007583
3: -0.0026804, -0.0013700, -0.0026927, -0.0013934, -0.0007393, 0.0007843
4: -0.0025539, -0.0011353, -0.0025286, -0.0011219, -0.0008491, 0.0008003
5: 0.0051877, 0.0065302, 0.0051751, 0.0065062, -0.0007574, 0.0008035
6: -0.0017169, 0.0036095, -0.0017671, 0.0035144, -0.0030051, 0.0031881
7: -0.0074726, -0.0002184, -0.0073431, -0.0001500, -0.0043419, 0.0040927
8: 0.9839500, 0.9890600, 0.9840412, 0.9891082, -0.0030585, 0.0028829
9: -0.0059567, -0.0013182, -0.0060004, -0.0014010, -0.0026170, 0.0027763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026952, 0.0049869, 0.0027971, 0.0050781, -0.0013293, 0.0010978
1: 0.0017117, 0.0020428, 0.0017264, 0.0020559, -0.0001921, 0.0001586
2: 0.0116027, 0.0128697, 0.0115523, 0.0128134, -0.0006069, 0.0007350
3: -0.0026804, -0.0013700, -0.0027325, -0.0014282, -0.0006277, 0.0007601
4: -0.0025539, -0.0011353, -0.0024908, -0.0010788, -0.0008229, 0.0006795
5: 0.0051877, 0.0065302, 0.0051343, 0.0064705, -0.0006431, 0.0007787
6: -0.0017169, 0.0036095, -0.0019289, 0.0033727, -0.0025515, 0.0030898
7: -0.0074726, -0.0002184, -0.0071500, 0.0000703, -0.0042080, 0.0034749
8: 0.9839500, 0.9890600, 0.9841773, 0.9892634, -0.0029642, 0.0024478
9: -0.0059567, -0.0013182, -0.0061413, -0.0015244, -0.0022220, 0.0026907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026581, 0.0050153, 0.0027361, 0.0050085, -0.0014676, 0.0013947
1: 0.0017063, 0.0020469, 0.0017176, 0.0020459, -0.0002120, 0.0002015
2: 0.0115870, 0.0128903, 0.0115908, 0.0128471, -0.0007711, 0.0008114
3: -0.0026966, -0.0013487, -0.0026927, -0.0013934, -0.0007975, 0.0008392
4: -0.0025769, -0.0011177, -0.0025286, -0.0011219, -0.0009085, 0.0008634
5: 0.0051711, 0.0065520, 0.0051751, 0.0065062, -0.0008170, 0.0008597
6: -0.0017830, 0.0036959, -0.0017671, 0.0035144, -0.0032417, 0.0034111
7: -0.0075902, -0.0001284, -0.0073431, -0.0001500, -0.0046456, 0.0044149
8: 0.9838672, 0.9891235, 0.9840412, 0.9891082, -0.0032725, 0.0031100
9: -0.0060143, -0.0012430, -0.0060004, -0.0014010, -0.0028230, 0.0029705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026581, 0.0050153, 0.0027971, 0.0050781, -0.0014507, 0.0012114
1: 0.0017063, 0.0020469, 0.0017264, 0.0020559, -0.0002096, 0.0001750
2: 0.0115870, 0.0128903, 0.0115523, 0.0128134, -0.0006697, 0.0008021
3: -0.0026966, -0.0013487, -0.0027325, -0.0014282, -0.0006927, 0.0008295
4: -0.0025769, -0.0011177, -0.0024908, -0.0010788, -0.0008980, 0.0007498
5: 0.0051711, 0.0065520, 0.0051343, 0.0064705, -0.0007096, 0.0008498
6: -0.0017830, 0.0036959, -0.0019289, 0.0033727, -0.0028155, 0.0033719
7: -0.0075902, -0.0001284, -0.0071500, 0.0000703, -0.0045922, 0.0038345
8: 0.9838672, 0.9891235, 0.9841773, 0.9892634, -0.0032348, 0.0027011
9: -0.0060143, -0.0012430, -0.0061413, -0.0015244, -0.0024519, 0.0029364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027065, 0.0050339, 0.0026126, 0.0049584, -0.0013544, 0.0015125
1: 0.0017133, 0.0020495, 0.0016997, 0.0020387, -0.0001957, 0.0002185
2: 0.0115768, 0.0128635, 0.0116185, 0.0129154, -0.0008362, 0.0007488
3: -0.0027072, -0.0013764, -0.0026641, -0.0013227, -0.0008649, 0.0007745
4: -0.0025469, -0.0011062, -0.0026051, -0.0011529, -0.0008384, 0.0009363
5: 0.0051602, 0.0065236, 0.0052044, 0.0065786, -0.0008860, 0.0007934
6: -0.0018261, 0.0035834, -0.0016508, 0.0038017, -0.0035156, 0.0031480
7: -0.0074370, -0.0000698, -0.0077342, -0.0003085, -0.0042873, 0.0047879
8: 0.9839751, 0.9891647, 0.9837657, 0.9889965, -0.0030201, 0.0033727
9: -0.0060517, -0.0013410, -0.0058991, -0.0011509, -0.0030615, 0.0027414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027065, 0.0050339, 0.0026683, 0.0050340, -0.0013106, 0.0012806
1: 0.0017133, 0.0020495, 0.0017078, 0.0020496, -0.0001893, 0.0001850
2: 0.0115768, 0.0128635, 0.0115767, 0.0128846, -0.0007080, 0.0007246
3: -0.0027072, -0.0013764, -0.0027073, -0.0013545, -0.0007323, 0.0007494
4: -0.0025469, -0.0011062, -0.0025706, -0.0011061, -0.0008113, 0.0007927
5: 0.0051602, 0.0065236, 0.0051601, 0.0065460, -0.0007502, 0.0007677
6: -0.0018261, 0.0035834, -0.0018265, 0.0036722, -0.0029765, 0.0030461
7: -0.0074370, -0.0000698, -0.0075580, -0.0000692, -0.0041485, 0.0040537
8: 0.9839751, 0.9891647, 0.9838899, 0.9891652, -0.0029223, 0.0028555
9: -0.0060517, -0.0013410, -0.0060521, -0.0012636, -0.0025921, 0.0026527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025791, 0.0049901, 0.0026126, 0.0049584, -0.0013697, 0.0013309
1: 0.0016949, 0.0020432, 0.0016997, 0.0020387, -0.0001979, 0.0001923
2: 0.0116009, 0.0129339, 0.0116185, 0.0129154, -0.0007358, 0.0007573
3: -0.0026822, -0.0013036, -0.0026641, -0.0013227, -0.0007610, 0.0007832
4: -0.0026258, -0.0011333, -0.0026051, -0.0011529, -0.0008479, 0.0008238
5: 0.0051858, 0.0065982, 0.0052044, 0.0065786, -0.0007796, 0.0008024
6: -0.0017244, 0.0038795, -0.0016508, 0.0038017, -0.0030933, 0.0031835
7: -0.0078402, -0.0002082, -0.0077342, -0.0003085, -0.0043357, 0.0042128
8: 0.9836910, 0.9890672, 0.9837657, 0.9889965, -0.0030542, 0.0029676
9: -0.0059632, -0.0010831, -0.0058991, -0.0011509, -0.0026938, 0.0027724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017804
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025791, 0.0049901, 0.0026683, 0.0050340, -0.0013310, 0.0011007
1: 0.0016949, 0.0020432, 0.0017078, 0.0020496, -0.0001923, 0.0001590
2: 0.0116009, 0.0129339, 0.0115767, 0.0128846, -0.0006085, 0.0007359
3: -0.0026822, -0.0013036, -0.0027073, -0.0013545, -0.0006294, 0.0007611
4: -0.0026258, -0.0011333, -0.0025706, -0.0011061, -0.0008239, 0.0006813
5: 0.0051858, 0.0065982, 0.0051601, 0.0065460, -0.0006448, 0.0007797
6: -0.0017244, 0.0038795, -0.0018265, 0.0036722, -0.0025583, 0.0030937
7: -0.0078402, -0.0002082, -0.0075580, -0.0000692, -0.0042133, 0.0034842
8: 0.9836910, 0.9890672, 0.9838899, 0.9891652, -0.0029679, 0.0024543
9: -0.0059632, -0.0010831, -0.0060521, -0.0012636, -0.0022279, 0.0026941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027149, 0.0050254, 0.0026989, 0.0050603, -0.0014275, 0.0014686
1: 0.0017145, 0.0020483, 0.0017122, 0.0020534, -0.0002062, 0.0002122
2: 0.0115814, 0.0128589, 0.0115622, 0.0128677, -0.0008120, 0.0007892
3: -0.0027024, -0.0013812, -0.0027223, -0.0013720, -0.0008398, 0.0008163
4: -0.0025417, -0.0011114, -0.0025516, -0.0010899, -0.0008837, 0.0009091
5: 0.0051652, 0.0065187, 0.0051447, 0.0065281, -0.0008603, 0.0008362
6: -0.0018065, 0.0035639, -0.0018875, 0.0036011, -0.0034134, 0.0033180
7: -0.0074105, -0.0000964, -0.0074611, 0.0000140, -0.0045188, 0.0046488
8: 0.9839938, 0.9891460, 0.9839581, 0.9892237, -0.0031832, 0.0032747
9: -0.0060347, -0.0013579, -0.0061053, -0.0013255, -0.0029726, 0.0028895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027149, 0.0050254, 0.0027509, 0.0051409, -0.0013650, 0.0012710
1: 0.0017145, 0.0020483, 0.0017197, 0.0020650, -0.0001972, 0.0001836
2: 0.0115814, 0.0128589, 0.0115176, 0.0128390, -0.0007027, 0.0007547
3: -0.0027024, -0.0013812, -0.0027684, -0.0014018, -0.0007268, 0.0007805
4: -0.0025417, -0.0011114, -0.0025194, -0.0010400, -0.0008450, 0.0007868
5: 0.0051652, 0.0065187, 0.0050975, 0.0064976, -0.0007445, 0.0007996
6: -0.0018065, 0.0035639, -0.0020748, 0.0034802, -0.0029542, 0.0031727
7: -0.0074105, -0.0000964, -0.0072965, 0.0002690, -0.0043209, 0.0040233
8: 0.9839938, 0.9891460, 0.9840741, 0.9894034, -0.0030438, 0.0028341
9: -0.0060347, -0.0013579, -0.0062684, -0.0014308, -0.0025726, 0.0027629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026735, 0.0050523, 0.0026989, 0.0050603, -0.0013259, 0.0013794
1: 0.0017085, 0.0020522, 0.0017122, 0.0020534, -0.0001915, 0.0001993
2: 0.0115666, 0.0128817, 0.0115622, 0.0128677, -0.0007626, 0.0007330
3: -0.0027178, -0.0013576, -0.0027223, -0.0013720, -0.0007888, 0.0007581
4: -0.0025673, -0.0010948, -0.0025516, -0.0010899, -0.0008207, 0.0008539
5: 0.0051494, 0.0065429, 0.0051447, 0.0065281, -0.0008081, 0.0007767
6: -0.0018690, 0.0036600, -0.0018875, 0.0036011, -0.0032061, 0.0030816
7: -0.0075413, -0.0000112, -0.0074611, 0.0000140, -0.0041969, 0.0043665
8: 0.9839016, 0.9892060, 0.9839581, 0.9892237, -0.0029564, 0.0030758
9: -0.0060892, -0.0012742, -0.0061053, -0.0013255, -0.0027920, 0.0026836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026735, 0.0050523, 0.0027509, 0.0051409, -0.0012664, 0.0011852
1: 0.0017085, 0.0020522, 0.0017197, 0.0020650, -0.0001830, 0.0001712
2: 0.0115666, 0.0128817, 0.0115176, 0.0128390, -0.0006553, 0.0007001
3: -0.0027178, -0.0013576, -0.0027684, -0.0014018, -0.0006777, 0.0007241
4: -0.0025673, -0.0010948, -0.0025194, -0.0010400, -0.0007839, 0.0007336
5: 0.0051494, 0.0065429, 0.0050975, 0.0064976, -0.0006943, 0.0007418
6: -0.0018690, 0.0036600, -0.0020748, 0.0034802, -0.0027547, 0.0029433
7: -0.0075413, -0.0000112, -0.0072965, 0.0002690, -0.0040086, 0.0037516
8: 0.9839016, 0.9892060, 0.9840741, 0.9894034, -0.0028237, 0.0026427
9: -0.0060892, -0.0012742, -0.0062684, -0.0014308, -0.0023989, 0.0025632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0027077, 0.0050711, 0.0025763, 0.0050099, -0.0014473, 0.0015435
1: 0.0017135, 0.0020549, 0.0016945, 0.0020461, -0.0002091, 0.0002230
2: 0.0115562, 0.0128629, 0.0115900, 0.0129355, -0.0008534, 0.0008002
3: -0.0027285, -0.0013771, -0.0026935, -0.0013020, -0.0008826, 0.0008276
4: -0.0025462, -0.0010832, -0.0026275, -0.0011211, -0.0008959, 0.0009555
5: 0.0051384, 0.0065229, 0.0051743, 0.0065998, -0.0009042, 0.0008478
6: -0.0019126, 0.0035806, -0.0017704, 0.0038858, -0.0035876, 0.0033639
7: -0.0074332, 0.0000480, -0.0078489, -0.0001455, -0.0045814, 0.0048860
8: 0.9839778, 0.9892477, 0.9836849, 0.9891114, -0.0032272, 0.0034418
9: -0.0061271, -0.0013433, -0.0060033, -0.0010775, -0.0031242, 0.0029295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018486
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018351
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0027077, 0.0050711, 0.0026326, 0.0050911, -0.0013986, 0.0013030
1: 0.0017135, 0.0020549, 0.0017026, 0.0020578, -0.0002021, 0.0001882
2: 0.0115562, 0.0128629, 0.0115451, 0.0129044, -0.0007204, 0.0007732
3: -0.0027285, -0.0013771, -0.0027400, -0.0013342, -0.0007450, 0.0007997
4: -0.0025462, -0.0010832, -0.0025926, -0.0010708, -0.0008657, 0.0008066
5: 0.0051384, 0.0065229, 0.0051267, 0.0065669, -0.0007633, 0.0008193
6: -0.0019126, 0.0035806, -0.0019592, 0.0037550, -0.0030284, 0.0032506
7: -0.0074332, 0.0000480, -0.0076708, 0.0001116, -0.0044271, 0.0041245
8: 0.9839778, 0.9892477, 0.9838104, 0.9892925, -0.0031185, 0.0029054
9: -0.0061271, -0.0013433, -0.0061677, -0.0011915, -0.0026373, 0.0028308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018486
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018351
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025873, 0.0050291, 0.0025763, 0.0050099, -0.0014587, 0.0013618
1: 0.0016961, 0.0020489, 0.0016945, 0.0020461, -0.0002107, 0.0001967
2: 0.0115794, 0.0129294, 0.0115900, 0.0129355, -0.0007529, 0.0008065
3: -0.0027045, -0.0013083, -0.0026935, -0.0013020, -0.0007787, 0.0008341
4: -0.0026207, -0.0011092, -0.0026275, -0.0011211, -0.0009029, 0.0008430
5: 0.0051630, 0.0065934, 0.0051743, 0.0065998, -0.0007977, 0.0008545
6: -0.0018151, 0.0038604, -0.0017704, 0.0038858, -0.0031652, 0.0033903
7: -0.0078143, -0.0000847, -0.0078489, -0.0001455, -0.0046173, 0.0043107
8: 0.9837093, 0.9891542, 0.9836849, 0.9891114, -0.0032525, 0.0030365
9: -0.0060422, -0.0010997, -0.0060033, -0.0010775, -0.0027564, 0.0029524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018102
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017949
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025873, 0.0050291, 0.0026326, 0.0050911, -0.0014096, 0.0011216
1: 0.0016961, 0.0020489, 0.0017026, 0.0020578, -0.0002036, 0.0001620
2: 0.0115794, 0.0129294, 0.0115451, 0.0129044, -0.0006201, 0.0007793
3: -0.0027045, -0.0013083, -0.0027400, -0.0013342, -0.0006413, 0.0008060
4: -0.0026207, -0.0011092, -0.0025926, -0.0010708, -0.0008726, 0.0006943
5: 0.0051630, 0.0065934, 0.0051267, 0.0065669, -0.0006570, 0.0008257
6: -0.0018151, 0.0038604, -0.0019592, 0.0037550, -0.0026068, 0.0032763
7: -0.0078143, -0.0000847, -0.0076708, 0.0001116, -0.0044621, 0.0035503
8: 0.9837093, 0.9891542, 0.9838104, 0.9892925, -0.0031432, 0.0025009
9: -0.0060422, -0.0010997, -0.0061677, -0.0011915, -0.0022701, 0.0028532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018102
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017949
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026952, 0.0049869, 0.0027022, 0.0050324, -0.0014866, 0.0014176
1: 0.0017117, 0.0020428, 0.0017127, 0.0020493, -0.0002148, 0.0002048
2: 0.0116027, 0.0128697, 0.0115776, 0.0128659, -0.0007838, 0.0008219
3: -0.0026804, -0.0013700, -0.0027064, -0.0013739, -0.0008106, 0.0008501
4: -0.0025539, -0.0011353, -0.0025496, -0.0011071, -0.0009202, 0.0008775
5: 0.0051877, 0.0065302, 0.0051611, 0.0065261, -0.0008304, 0.0008708
6: -0.0017169, 0.0036095, -0.0018227, 0.0035934, -0.0032950, 0.0034553
7: -0.0074726, -0.0002184, -0.0074506, -0.0000743, -0.0047058, 0.0044875
8: 0.9839500, 0.9890600, 0.9839656, 0.9891616, -0.0033149, 0.0031611
9: -0.0059567, -0.0013182, -0.0060488, -0.0013323, -0.0028694, 0.0030090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026952, 0.0049869, 0.0027543, 0.0051142, -0.0014461, 0.0012190
1: 0.0017117, 0.0020428, 0.0017202, 0.0020611, -0.0002089, 0.0001761
2: 0.0116027, 0.0128697, 0.0115324, 0.0128371, -0.0006740, 0.0007995
3: -0.0026804, -0.0013700, -0.0027531, -0.0014037, -0.0006970, 0.0008269
4: -0.0025539, -0.0011353, -0.0025173, -0.0010565, -0.0008952, 0.0007546
5: 0.0051877, 0.0065302, 0.0051132, 0.0064956, -0.0007141, 0.0008471
6: -0.0017169, 0.0036095, -0.0020127, 0.0034723, -0.0028333, 0.0033612
7: -0.0074726, -0.0002184, -0.0072857, 0.0001844, -0.0045776, 0.0038587
8: 0.9839500, 0.9890600, 0.9840816, 0.9893438, -0.0032246, 0.0027182
9: -0.0059567, -0.0013182, -0.0062143, -0.0014377, -0.0024674, 0.0029271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017759
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0026581, 0.0050153, 0.0027022, 0.0050324, -0.0013895, 0.0013169
1: 0.0017063, 0.0020469, 0.0017127, 0.0020493, -0.0002007, 0.0001903
2: 0.0115870, 0.0128903, 0.0115776, 0.0128659, -0.0007281, 0.0007682
3: -0.0026966, -0.0013487, -0.0027064, -0.0013739, -0.0007530, 0.0007945
4: -0.0025769, -0.0011177, -0.0025496, -0.0011071, -0.0008601, 0.0008152
5: 0.0051711, 0.0065520, 0.0051611, 0.0065261, -0.0007715, 0.0008140
6: -0.0017830, 0.0036959, -0.0018227, 0.0035934, -0.0030609, 0.0032295
7: -0.0075902, -0.0001284, -0.0074506, -0.0000743, -0.0043983, 0.0041687
8: 0.9838672, 0.9891235, 0.9839656, 0.9891616, -0.0030983, 0.0029365
9: -0.0060143, -0.0012430, -0.0060488, -0.0013323, -0.0026656, 0.0028124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0026581, 0.0050153, 0.0027543, 0.0051142, -0.0013509, 0.0011219
1: 0.0017063, 0.0020469, 0.0017202, 0.0020611, -0.0001952, 0.0001621
2: 0.0115870, 0.0128903, 0.0115324, 0.0128371, -0.0006203, 0.0007469
3: -0.0026966, -0.0013487, -0.0027531, -0.0014037, -0.0006415, 0.0007725
4: -0.0025769, -0.0011177, -0.0025173, -0.0010565, -0.0008362, 0.0006945
5: 0.0051711, 0.0065520, 0.0051132, 0.0064956, -0.0006572, 0.0007914
6: -0.0017830, 0.0036959, -0.0020127, 0.0034723, -0.0026076, 0.0031399
7: -0.0075902, -0.0001284, -0.0072857, 0.0001844, -0.0042763, 0.0035513
8: 0.9838672, 0.9891235, 0.9840816, 0.9893438, -0.0030123, 0.0025016
9: -0.0060143, -0.0012430, -0.0062143, -0.0014377, -0.0022708, 0.0027344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0026887, 0.0050345, 0.0025776, 0.0049785, -0.0015024, 0.0015281
1: 0.0017107, 0.0020496, 0.0016947, 0.0020416, -0.0002170, 0.0002208
2: 0.0115764, 0.0128733, 0.0116074, 0.0129348, -0.0008449, 0.0008306
3: -0.0027076, -0.0013663, -0.0026756, -0.0013027, -0.0008738, 0.0008591
4: -0.0025579, -0.0011059, -0.0026267, -0.0011405, -0.0009300, 0.0009459
5: 0.0051599, 0.0065340, 0.0051926, 0.0065991, -0.0008952, 0.0008801
6: -0.0018275, 0.0036246, -0.0016974, 0.0038830, -0.0035518, 0.0034919
7: -0.0074931, -0.0000679, -0.0078450, -0.0002449, -0.0047556, 0.0048373
8: 0.9839355, 0.9891660, 0.9836877, 0.9890413, -0.0033500, 0.0034075
9: -0.0060529, -0.0013050, -0.0059397, -0.0010801, -0.0030931, 0.0030409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018273
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018173
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0026887, 0.0050345, 0.0026339, 0.0050628, -0.0014761, 0.0012974
1: 0.0017107, 0.0020496, 0.0017028, 0.0020537, -0.0002133, 0.0001874
2: 0.0115764, 0.0128733, 0.0115608, 0.0129037, -0.0007173, 0.0008161
3: -0.0027076, -0.0013663, -0.0027237, -0.0013349, -0.0007419, 0.0008441
4: -0.0025579, -0.0011059, -0.0025919, -0.0010883, -0.0009138, 0.0008031
5: 0.0051599, 0.0065340, 0.0051433, 0.0065661, -0.0007600, 0.0008647
6: -0.0018275, 0.0036246, -0.0018932, 0.0037521, -0.0030155, 0.0034310
7: -0.0074931, -0.0000679, -0.0076667, 0.0000217, -0.0046727, 0.0041069
8: 0.9839355, 0.9891660, 0.9838133, 0.9892291, -0.0032915, 0.0028930
9: -0.0060529, -0.0013050, -0.0061102, -0.0011940, -0.0026261, 0.0029878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018273
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018173
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0025661, 0.0049907, 0.0025776, 0.0049785, -0.0015268, 0.0013461
1: 0.0016930, 0.0020433, 0.0016947, 0.0020416, -0.0002206, 0.0001945
2: 0.0116006, 0.0129411, 0.0116074, 0.0129348, -0.0007442, 0.0008441
3: -0.0026825, -0.0012961, -0.0026756, -0.0013027, -0.0007697, 0.0008730
4: -0.0026338, -0.0011330, -0.0026267, -0.0011405, -0.0009451, 0.0008333
5: 0.0051855, 0.0066058, 0.0051926, 0.0065991, -0.0007886, 0.0008944
6: -0.0017258, 0.0039097, -0.0016974, 0.0038830, -0.0031288, 0.0035486
7: -0.0078814, -0.0002064, -0.0078450, -0.0002449, -0.0048329, 0.0042611
8: 0.9836620, 0.9890685, 0.9836877, 0.9890413, -0.0034044, 0.0030016
9: -0.0059644, -0.0010567, -0.0059397, -0.0010801, -0.0027247, 0.0030903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017981
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017831
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0025661, 0.0049907, 0.0026339, 0.0050628, -0.0015006, 0.0011153
1: 0.0016930, 0.0020433, 0.0017028, 0.0020537, -0.0002168, 0.0001611
2: 0.0116006, 0.0129411, 0.0115608, 0.0129037, -0.0006166, 0.0008297
3: -0.0026825, -0.0012961, -0.0027237, -0.0013349, -0.0006377, 0.0008581
4: -0.0026338, -0.0011330, -0.0025919, -0.0010883, -0.0009289, 0.0006904
5: 0.0051855, 0.0066058, 0.0051433, 0.0065661, -0.0006533, 0.0008791
6: -0.0017258, 0.0039097, -0.0018932, 0.0037521, -0.0025922, 0.0034879
7: -0.0078814, -0.0002064, -0.0076667, 0.0000217, -0.0047502, 0.0035303
8: 0.9836620, 0.9890685, 0.9838133, 0.9892291, -0.0033462, 0.0024868
9: -0.0059644, -0.0010567, -0.0061102, -0.0011940, -0.0022574, 0.0030374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017981
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017831
time: 1.06 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.38 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018149
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018149
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017968
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017968
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0017872
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0017872
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018291
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018291
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018023
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018023
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017675
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017675
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017618
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017617
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017849
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017848
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018621
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018493
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018621
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018493
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018059
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017934
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018264
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018154
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018545
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018445
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018545
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018445
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017919
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017766
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0018146
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017999
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0017979
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0017858
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018395
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018288
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017948
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017808
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018156
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018035
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017854
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017700
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018362
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018286
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017799
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017639
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0018030
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017877
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017804
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018486
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018351
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018486
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018351
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018102
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017949
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018102
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017949
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017759
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018273
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018173
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018273
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018173
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017981
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017831
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017981
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017831

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027382, 0.0049981, 0.0026542, 0.0049498, -0.0011273, 0.0011606
1: 0.0017179, 0.0020444, 0.0017058, 0.0020374, -0.0001629, 0.0001677
2: 0.0115965, 0.0128460, 0.0116232, 0.0128924, -0.0006416, 0.0006232
3: -0.0026868, -0.0013945, -0.0026592, -0.0013465, -0.0006636, 0.0006446
4: -0.0025273, -0.0011284, -0.0025793, -0.0011583, -0.0006978, 0.0007184
5: 0.0051812, 0.0065050, 0.0052095, 0.0065542, -0.0006799, 0.0006603
6: -0.0017430, 0.0035097, -0.0016307, 0.0037050, -0.0026974, 0.0026200
7: -0.0073365, -0.0001829, -0.0076026, -0.0003358, -0.0035683, 0.0036737
8: 0.9840458, 0.9890851, 0.9838585, 0.9889773, -0.0025136, 0.0025878
9: -0.0059794, -0.0014052, -0.0058816, -0.0012351, -0.0023491, 0.0022816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018053
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018521
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0027043, 0.0050226, 0.0026542, 0.0049498, -0.0012382, 0.0012781
1: 0.0017130, 0.0020479, 0.0017058, 0.0020374, -0.0001789, 0.0001846
2: 0.0115830, 0.0128647, 0.0116232, 0.0128924, -0.0007066, 0.0006846
3: -0.0027008, -0.0013751, -0.0026592, -0.0013465, -0.0007308, 0.0007080
4: -0.0025483, -0.0011132, -0.0025793, -0.0011583, -0.0007664, 0.0007911
5: 0.0051668, 0.0065249, 0.0052095, 0.0065542, -0.0007487, 0.0007253
6: -0.0017999, 0.0035885, -0.0016307, 0.0037050, -0.0029706, 0.0028778
7: -0.0074440, -0.0001054, -0.0076026, -0.0003358, -0.0039194, 0.0040456
8: 0.9839701, 0.9891396, 0.9838585, 0.9889773, -0.0027609, 0.0028498
9: -0.0060289, -0.0013365, -0.0058816, -0.0012351, -0.0025869, 0.0025062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018053
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018521
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027382, 0.0049981, 0.0026334, 0.0049106, -0.0011262, 0.0012447
1: 0.0017179, 0.0020444, 0.0017028, 0.0020317, -0.0001627, 0.0001798
2: 0.0115965, 0.0128460, 0.0116449, 0.0129039, -0.0006881, 0.0006227
3: -0.0026868, -0.0013945, -0.0026368, -0.0013346, -0.0007117, 0.0006440
4: -0.0025273, -0.0011284, -0.0025921, -0.0011825, -0.0006972, 0.0007705
5: 0.0051812, 0.0065050, 0.0052324, 0.0065664, -0.0007291, 0.0006598
6: -0.0017430, 0.0035097, -0.0015397, 0.0037532, -0.0028929, 0.0026177
7: -0.0073365, -0.0001829, -0.0076683, -0.0004598, -0.0035651, 0.0039399
8: 0.9840458, 0.9890851, 0.9838122, 0.9888900, -0.0025113, 0.0027754
9: -0.0059794, -0.0014052, -0.0058024, -0.0011931, -0.0025193, 0.0022796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018020
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0027043, 0.0050226, 0.0026334, 0.0049106, -0.0012372, 0.0013622
1: 0.0017130, 0.0020479, 0.0017028, 0.0020317, -0.0001787, 0.0001968
2: 0.0115830, 0.0128647, 0.0116449, 0.0129039, -0.0007531, 0.0006840
3: -0.0027008, -0.0013751, -0.0026368, -0.0013346, -0.0007789, 0.0007074
4: -0.0025483, -0.0011132, -0.0025921, -0.0011825, -0.0007658, 0.0008432
5: 0.0051668, 0.0065249, 0.0052324, 0.0065664, -0.0007980, 0.0007247
6: -0.0017999, 0.0035885, -0.0015397, 0.0037532, -0.0031660, 0.0028755
7: -0.0074440, -0.0001054, -0.0076683, -0.0004598, -0.0039162, 0.0043119
8: 0.9839701, 0.9891396, 0.9838122, 0.9888900, -0.0027586, 0.0030374
9: -0.0060289, -0.0013365, -0.0058024, -0.0011931, -0.0027571, 0.0025041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018020
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026168, 0.0049505, 0.0027350, 0.0050367, -0.0012873, 0.0010781
1: 0.0017003, 0.0020375, 0.0017174, 0.0020500, -0.0001860, 0.0001558
2: 0.0116229, 0.0129131, 0.0115752, 0.0128478, -0.0005961, 0.0007117
3: -0.0026596, -0.0013251, -0.0027088, -0.0013927, -0.0006165, 0.0007361
4: -0.0026025, -0.0011578, -0.0025293, -0.0011045, -0.0007969, 0.0006674
5: 0.0052091, 0.0065762, 0.0051586, 0.0065069, -0.0006315, 0.0007541
6: -0.0016323, 0.0037919, -0.0018326, 0.0035171, -0.0025058, 0.0029921
7: -0.0077210, -0.0003336, -0.0073467, -0.0000608, -0.0040750, 0.0034127
8: 0.9837750, 0.9889789, 0.9840387, 0.9891710, -0.0028705, 0.0024040
9: -0.0058830, -0.0011593, -0.0060574, -0.0013987, -0.0021822, 0.0026057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025818, 0.0049732, 0.0027350, 0.0050367, -0.0014000, 0.0011816
1: 0.0016953, 0.0020408, 0.0017174, 0.0020500, -0.0002023, 0.0001707
2: 0.0116103, 0.0129324, 0.0115752, 0.0128478, -0.0006533, 0.0007740
3: -0.0026725, -0.0013051, -0.0027088, -0.0013927, -0.0006757, 0.0008006
4: -0.0026241, -0.0011438, -0.0025293, -0.0011045, -0.0008666, 0.0007315
5: 0.0051957, 0.0065966, 0.0051586, 0.0065069, -0.0006922, 0.0008201
6: -0.0016851, 0.0038732, -0.0018326, 0.0035171, -0.0027465, 0.0032541
7: -0.0078316, -0.0002617, -0.0073467, -0.0000608, -0.0044318, 0.0037405
8: 0.9836971, 0.9890295, 0.9840387, 0.9891710, -0.0031218, 0.0026349
9: -0.0059290, -0.0010886, -0.0060574, -0.0013987, -0.0023917, 0.0028338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026168, 0.0049505, 0.0026114, 0.0049882, -0.0011049, 0.0010633
1: 0.0017003, 0.0020375, 0.0016996, 0.0020429, -0.0001596, 0.0001536
2: 0.0116229, 0.0129131, 0.0116020, 0.0129161, -0.0005879, 0.0006109
3: -0.0026596, -0.0013251, -0.0026811, -0.0013220, -0.0006080, 0.0006318
4: -0.0026025, -0.0011578, -0.0026058, -0.0011345, -0.0006840, 0.0006582
5: 0.0052091, 0.0065762, 0.0051870, 0.0065793, -0.0006229, 0.0006472
6: -0.0016323, 0.0037919, -0.0017199, 0.0038044, -0.0024714, 0.0025681
7: -0.0077210, -0.0003336, -0.0077380, -0.0002144, -0.0034975, 0.0033659
8: 0.9837750, 0.9889789, 0.9837630, 0.9890628, -0.0024637, 0.0023710
9: -0.0058830, -0.0011593, -0.0059593, -0.0011485, -0.0021522, 0.0022364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018149
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018149
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025818, 0.0049732, 0.0026114, 0.0049882, -0.0012304, 0.0011824
1: 0.0016953, 0.0020408, 0.0016996, 0.0020429, -0.0001778, 0.0001708
2: 0.0116103, 0.0129324, 0.0116020, 0.0129161, -0.0006537, 0.0006803
3: -0.0026725, -0.0013051, -0.0026811, -0.0013220, -0.0006761, 0.0007036
4: -0.0026241, -0.0011438, -0.0026058, -0.0011345, -0.0007617, 0.0007319
5: 0.0051957, 0.0065966, 0.0051870, 0.0065793, -0.0006926, 0.0007208
6: -0.0016851, 0.0038732, -0.0017199, 0.0038044, -0.0027481, 0.0028599
7: -0.0078316, -0.0002617, -0.0077380, -0.0002144, -0.0038949, 0.0037427
8: 0.9836971, 0.9890295, 0.9837630, 0.9890628, -0.0027436, 0.0026364
9: -0.0059290, -0.0010886, -0.0059593, -0.0011485, -0.0023932, 0.0024905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018149
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018149
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027382, 0.0049981, 0.0027149, 0.0050254, -0.0013525, 0.0012594
1: 0.0017179, 0.0020444, 0.0017145, 0.0020483, -0.0001954, 0.0001819
2: 0.0115965, 0.0128460, 0.0115814, 0.0128589, -0.0006963, 0.0007478
3: -0.0026868, -0.0013945, -0.0027024, -0.0013812, -0.0007201, 0.0007734
4: -0.0025273, -0.0011284, -0.0025417, -0.0011114, -0.0008372, 0.0007796
5: 0.0051812, 0.0065050, 0.0051652, 0.0065187, -0.0007378, 0.0007923
6: -0.0017430, 0.0035097, -0.0018065, 0.0035639, -0.0029272, 0.0031436
7: -0.0073365, -0.0001829, -0.0074105, -0.0000964, -0.0042814, 0.0039866
8: 0.9840458, 0.9890851, 0.9839938, 0.9891460, -0.0030159, 0.0028083
9: -0.0059794, -0.0014052, -0.0060347, -0.0013579, -0.0025492, 0.0027376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0017860
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018437
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0027043, 0.0050226, 0.0027149, 0.0050254, -0.0014634, 0.0013769
1: 0.0017130, 0.0020479, 0.0017145, 0.0020483, -0.0002114, 0.0001989
2: 0.0115830, 0.0128647, 0.0115814, 0.0128589, -0.0007613, 0.0008091
3: -0.0027008, -0.0013751, -0.0027024, -0.0013812, -0.0007873, 0.0008368
4: -0.0025483, -0.0011132, -0.0025417, -0.0011114, -0.0009059, 0.0008523
5: 0.0051668, 0.0065249, 0.0051652, 0.0065187, -0.0008066, 0.0008573
6: -0.0017999, 0.0035885, -0.0018065, 0.0035639, -0.0032003, 0.0034014
7: -0.0074440, -0.0001054, -0.0074105, -0.0000964, -0.0046325, 0.0043586
8: 0.9839701, 0.9891396, 0.9839938, 0.9891460, -0.0032632, 0.0030703
9: -0.0060289, -0.0013365, -0.0060347, -0.0013579, -0.0027870, 0.0029621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0017860
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018437
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027382, 0.0049981, 0.0026952, 0.0049869, -0.0013265, 0.0013164
1: 0.0017179, 0.0020444, 0.0017117, 0.0020428, -0.0001916, 0.0001902
2: 0.0115965, 0.0128460, 0.0116027, 0.0128697, -0.0007278, 0.0007334
3: -0.0026868, -0.0013945, -0.0026804, -0.0013700, -0.0007527, 0.0007585
4: -0.0025273, -0.0011284, -0.0025539, -0.0011353, -0.0008211, 0.0008149
5: 0.0051812, 0.0065050, 0.0051877, 0.0065302, -0.0007712, 0.0007771
6: -0.0017430, 0.0035097, -0.0017169, 0.0036095, -0.0030597, 0.0030832
7: -0.0073365, -0.0001829, -0.0074726, -0.0002184, -0.0041991, 0.0041671
8: 0.9840458, 0.9890851, 0.9839500, 0.9890600, -0.0029579, 0.0029354
9: -0.0059794, -0.0014052, -0.0059567, -0.0013182, -0.0026646, 0.0026850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017835
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0027043, 0.0050226, 0.0026952, 0.0049869, -0.0014374, 0.0014339
1: 0.0017130, 0.0020479, 0.0017117, 0.0020428, -0.0002077, 0.0002072
2: 0.0115830, 0.0128647, 0.0116027, 0.0128697, -0.0007928, 0.0007947
3: -0.0027008, -0.0013751, -0.0026804, -0.0013700, -0.0008199, 0.0008219
4: -0.0025483, -0.0011132, -0.0025539, -0.0011353, -0.0008898, 0.0008876
5: 0.0051668, 0.0065249, 0.0051877, 0.0065302, -0.0008400, 0.0008421
6: -0.0017999, 0.0035885, -0.0017169, 0.0036095, -0.0033329, 0.0033410
7: -0.0074440, -0.0001054, -0.0074726, -0.0002184, -0.0045502, 0.0045391
8: 0.9839701, 0.9891396, 0.9839500, 0.9890600, -0.0032052, 0.0031974
9: -0.0060289, -0.0013365, -0.0059567, -0.0013182, -0.0029024, 0.0029095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017835
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026168, 0.0049505, 0.0027959, 0.0051052, -0.0015634, 0.0011816
1: 0.0017003, 0.0020375, 0.0017262, 0.0020599, -0.0002259, 0.0001707
2: 0.0116229, 0.0129131, 0.0115373, 0.0128141, -0.0006533, 0.0008643
3: -0.0026596, -0.0013251, -0.0027480, -0.0014275, -0.0006756, 0.0008939
4: -0.0026025, -0.0011578, -0.0024916, -0.0010620, -0.0009677, 0.0007314
5: 0.0052091, 0.0065762, 0.0051184, 0.0064712, -0.0006922, 0.0009158
6: -0.0016323, 0.0037919, -0.0019920, 0.0033755, -0.0027463, 0.0036337
7: -0.0077210, -0.0003336, -0.0071539, 0.0001562, -0.0049488, 0.0037402
8: 0.9837750, 0.9889789, 0.9841745, 0.9893240, -0.0034860, 0.0026347
9: -0.0058830, -0.0011593, -0.0061962, -0.0015220, -0.0023916, 0.0031644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025818, 0.0049732, 0.0027959, 0.0051052, -0.0016761, 0.0012851
1: 0.0016953, 0.0020408, 0.0017262, 0.0020599, -0.0002421, 0.0001857
2: 0.0116103, 0.0129324, 0.0115373, 0.0128141, -0.0007105, 0.0009266
3: -0.0026725, -0.0013051, -0.0027480, -0.0014275, -0.0007348, 0.0009584
4: -0.0026241, -0.0011438, -0.0024916, -0.0010620, -0.0010375, 0.0007955
5: 0.0051957, 0.0065966, 0.0051184, 0.0064712, -0.0007528, 0.0009818
6: -0.0016851, 0.0038732, -0.0019920, 0.0033755, -0.0029870, 0.0038956
7: -0.0078316, -0.0002617, -0.0071539, 0.0001562, -0.0053055, 0.0040680
8: 0.9836971, 0.9890295, 0.9841745, 0.9893240, -0.0037373, 0.0028656
9: -0.0059290, -0.0010886, -0.0061962, -0.0015220, -0.0026012, 0.0033925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026168, 0.0049505, 0.0026667, 0.0050627, -0.0013814, 0.0011712
1: 0.0017003, 0.0020375, 0.0017076, 0.0020537, -0.0001996, 0.0001692
2: 0.0116229, 0.0129131, 0.0115608, 0.0128855, -0.0006476, 0.0007638
3: -0.0026596, -0.0013251, -0.0027237, -0.0013537, -0.0006697, 0.0007899
4: -0.0026025, -0.0011578, -0.0025715, -0.0010884, -0.0008551, 0.0007250
5: 0.0052091, 0.0065762, 0.0051433, 0.0065469, -0.0006861, 0.0008092
6: -0.0016323, 0.0037919, -0.0018931, 0.0036757, -0.0027223, 0.0032108
7: -0.0077210, -0.0003336, -0.0075627, 0.0000215, -0.0043729, 0.0037075
8: 0.9837750, 0.9889789, 0.9838865, 0.9892291, -0.0030803, 0.0026117
9: -0.0058830, -0.0011593, -0.0061101, -0.0012605, -0.0023707, 0.0027961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017968
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017968
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025818, 0.0049732, 0.0026667, 0.0050627, -0.0015070, 0.0012903
1: 0.0016953, 0.0020408, 0.0017076, 0.0020537, -0.0002177, 0.0001864
2: 0.0116103, 0.0129324, 0.0115608, 0.0128855, -0.0007134, 0.0008332
3: -0.0026725, -0.0013051, -0.0027237, -0.0013537, -0.0007378, 0.0008617
4: -0.0026241, -0.0011438, -0.0025715, -0.0010884, -0.0009328, 0.0007987
5: 0.0051957, 0.0065966, 0.0051433, 0.0065469, -0.0007559, 0.0008828
6: -0.0016851, 0.0038732, -0.0018931, 0.0036757, -0.0029990, 0.0035026
7: -0.0078316, -0.0002617, -0.0075627, 0.0000215, -0.0047702, 0.0040844
8: 0.9836971, 0.9890295, 0.9838865, 0.9892291, -0.0033603, 0.0028771
9: -0.0059290, -0.0010886, -0.0061101, -0.0012605, -0.0026117, 0.0030502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017968
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017968
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027166, 0.0049629, 0.0027361, 0.0050085, -0.0012311, 0.0010658
1: 0.0017148, 0.0020393, 0.0017176, 0.0020459, -0.0001779, 0.0001540
2: 0.0116160, 0.0128579, 0.0115908, 0.0128471, -0.0005892, 0.0006806
3: -0.0026666, -0.0013822, -0.0026927, -0.0013934, -0.0006094, 0.0007039
4: -0.0025406, -0.0011502, -0.0025286, -0.0011219, -0.0007621, 0.0006597
5: 0.0052018, 0.0065176, 0.0051751, 0.0065062, -0.0006243, 0.0007212
6: -0.0016611, 0.0035598, -0.0017671, 0.0035144, -0.0024771, 0.0028613
7: -0.0074048, -0.0002945, -0.0073431, -0.0001500, -0.0038969, 0.0033736
8: 0.9839978, 0.9890064, 0.9840412, 0.9891082, -0.0027451, 0.0023764
9: -0.0059080, -0.0013615, -0.0060004, -0.0014010, -0.0021572, 0.0024918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017872
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017872
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026846, 0.0049872, 0.0027361, 0.0050085, -0.0013472, 0.0011898
1: 0.0017101, 0.0020428, 0.0017176, 0.0020459, -0.0001946, 0.0001719
2: 0.0116026, 0.0128756, 0.0115908, 0.0128471, -0.0006578, 0.0007448
3: -0.0026806, -0.0013639, -0.0026927, -0.0013934, -0.0006803, 0.0007703
4: -0.0025605, -0.0011351, -0.0025286, -0.0011219, -0.0008339, 0.0007365
5: 0.0051875, 0.0065364, 0.0051751, 0.0065062, -0.0006970, 0.0007892
6: -0.0017177, 0.0036342, -0.0017671, 0.0035144, -0.0027654, 0.0031312
7: -0.0075062, -0.0002174, -0.0073431, -0.0001500, -0.0042644, 0.0037663
8: 0.9839263, 0.9890608, 0.9840412, 0.9891082, -0.0030040, 0.0026530
9: -0.0059574, -0.0012967, -0.0060004, -0.0014010, -0.0024082, 0.0027268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017872
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017872
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027166, 0.0049629, 0.0026126, 0.0049584, -0.0012248, 0.0012423
1: 0.0017148, 0.0020393, 0.0016997, 0.0020387, -0.0001770, 0.0001795
2: 0.0116160, 0.0128579, 0.0116185, 0.0129154, -0.0006868, 0.0006772
3: -0.0026666, -0.0013822, -0.0026641, -0.0013227, -0.0007103, 0.0007004
4: -0.0025406, -0.0011502, -0.0026051, -0.0011529, -0.0007582, 0.0007690
5: 0.0052018, 0.0065176, 0.0052044, 0.0065786, -0.0007277, 0.0007175
6: -0.0016611, 0.0035598, -0.0016508, 0.0038017, -0.0028874, 0.0028469
7: -0.0074048, -0.0002945, -0.0077342, -0.0003085, -0.0038772, 0.0039323
8: 0.9839978, 0.9890064, 0.9837657, 0.9889965, -0.0027312, 0.0027700
9: -0.0059080, -0.0013615, -0.0058991, -0.0011509, -0.0025144, 0.0024792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018287
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018287
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026846, 0.0049872, 0.0026126, 0.0049584, -0.0013410, 0.0013663
1: 0.0017101, 0.0020428, 0.0016997, 0.0020387, -0.0001937, 0.0001974
2: 0.0116026, 0.0128756, 0.0116185, 0.0129154, -0.0007554, 0.0007414
3: -0.0026806, -0.0013639, -0.0026641, -0.0013227, -0.0007813, 0.0007668
4: -0.0025605, -0.0011351, -0.0026051, -0.0011529, -0.0008301, 0.0008458
5: 0.0051875, 0.0065364, 0.0052044, 0.0065786, -0.0008004, 0.0007855
6: -0.0017177, 0.0036342, -0.0016508, 0.0038017, -0.0031757, 0.0031168
7: -0.0075062, -0.0002174, -0.0077342, -0.0003085, -0.0042447, 0.0043250
8: 0.9839263, 0.9890608, 0.9837657, 0.9889965, -0.0029901, 0.0030466
9: -0.0059574, -0.0012967, -0.0058991, -0.0011509, -0.0027655, 0.0027142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018287
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018287
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0025966, 0.0049114, 0.0027361, 0.0050085, -0.0013638, 0.0010688
1: 0.0016974, 0.0020319, 0.0017176, 0.0020459, -0.0001970, 0.0001544
2: 0.0116445, 0.0129243, 0.0115908, 0.0128471, -0.0005909, 0.0007540
3: -0.0026372, -0.0013136, -0.0026927, -0.0013934, -0.0006111, 0.0007799
4: -0.0026149, -0.0011820, -0.0025286, -0.0011219, -0.0008442, 0.0006616
5: 0.0052319, 0.0065880, 0.0051751, 0.0065062, -0.0006261, 0.0007989
6: -0.0015415, 0.0038388, -0.0017671, 0.0035144, -0.0024842, 0.0031700
7: -0.0077848, -0.0004573, -0.0073431, -0.0001500, -0.0043172, 0.0033832
8: 0.9837301, 0.9888917, 0.9840412, 0.9891082, -0.0030411, 0.0023832
9: -0.0058039, -0.0011185, -0.0060004, -0.0014010, -0.0021633, 0.0027605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025640, 0.0049310, 0.0027361, 0.0050085, -0.0014807, 0.0011779
1: 0.0016927, 0.0020347, 0.0017176, 0.0020459, -0.0002139, 0.0001702
2: 0.0116336, 0.0129423, 0.0115908, 0.0128471, -0.0006512, 0.0008186
3: -0.0026484, -0.0012949, -0.0026927, -0.0013934, -0.0006735, 0.0008466
4: -0.0026351, -0.0011699, -0.0025286, -0.0011219, -0.0009165, 0.0007291
5: 0.0052205, 0.0066070, 0.0051751, 0.0065062, -0.0006900, 0.0008674
6: -0.0015870, 0.0039145, -0.0017671, 0.0035144, -0.0027377, 0.0034414
7: -0.0078879, -0.0003954, -0.0073431, -0.0001500, -0.0046869, 0.0037285
8: 0.9836575, 0.9889354, 0.9840412, 0.9891082, -0.0033016, 0.0026264
9: -0.0058435, -0.0010526, -0.0060004, -0.0014010, -0.0023841, 0.0029970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0025966, 0.0049114, 0.0026126, 0.0049584, -0.0012100, 0.0010596
1: 0.0016974, 0.0020319, 0.0016997, 0.0020387, -0.0001748, 0.0001531
2: 0.0116445, 0.0129243, 0.0116185, 0.0129154, -0.0005858, 0.0006690
3: -0.0026372, -0.0013136, -0.0026641, -0.0013227, -0.0006059, 0.0006919
4: -0.0026149, -0.0011820, -0.0026051, -0.0011529, -0.0007490, 0.0006559
5: 0.0052319, 0.0065880, 0.0052044, 0.0065786, -0.0006207, 0.0007088
6: -0.0015415, 0.0038388, -0.0016508, 0.0038017, -0.0024627, 0.0028123
7: -0.0077848, -0.0004573, -0.0077342, -0.0003085, -0.0038302, 0.0033540
8: 0.9837301, 0.9888917, 0.9837657, 0.9889965, -0.0026980, 0.0023627
9: -0.0058039, -0.0011185, -0.0058991, -0.0011509, -0.0021447, 0.0024491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018439, upper bound: 0.0018023
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018439, upper bound: 0.0018023
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025640, 0.0049310, 0.0026126, 0.0049584, -0.0013308, 0.0011856
1: 0.0016927, 0.0020347, 0.0016997, 0.0020387, -0.0001923, 0.0001713
2: 0.0116336, 0.0129423, 0.0116185, 0.0129154, -0.0006555, 0.0007358
3: -0.0026484, -0.0012949, -0.0026641, -0.0013227, -0.0006779, 0.0007610
4: -0.0026351, -0.0011699, -0.0026051, -0.0011529, -0.0008238, 0.0007339
5: 0.0052205, 0.0066070, 0.0052044, 0.0065786, -0.0006945, 0.0007796
6: -0.0015870, 0.0039145, -0.0016508, 0.0038017, -0.0027557, 0.0030932
7: -0.0078879, -0.0003954, -0.0077342, -0.0003085, -0.0042127, 0.0037530
8: 0.9836575, 0.9889354, 0.9837657, 0.9889965, -0.0029675, 0.0026437
9: -0.0058435, -0.0010526, -0.0058991, -0.0011509, -0.0023998, 0.0026937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018439, upper bound: 0.0018023
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018439, upper bound: 0.0018023
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027166, 0.0049629, 0.0027971, 0.0050781, -0.0014857, 0.0011691
1: 0.0017148, 0.0020393, 0.0017264, 0.0020559, -0.0002146, 0.0001689
2: 0.0116160, 0.0128579, 0.0115523, 0.0128134, -0.0006464, 0.0008214
3: -0.0026666, -0.0013822, -0.0027325, -0.0014282, -0.0006685, 0.0008495
4: -0.0025406, -0.0011502, -0.0024908, -0.0010788, -0.0009197, 0.0007237
5: 0.0052018, 0.0065176, 0.0051343, 0.0064705, -0.0006849, 0.0008703
6: -0.0016611, 0.0035598, -0.0019289, 0.0033727, -0.0027173, 0.0034531
7: -0.0074048, -0.0002945, -0.0071500, 0.0000703, -0.0047029, 0.0037008
8: 0.9839978, 0.9890064, 0.9841773, 0.9892634, -0.0033128, 0.0026069
9: -0.0059080, -0.0013615, -0.0061413, -0.0015244, -0.0023664, 0.0030072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017826, upper bound: 0.0017675
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017826, upper bound: 0.0017675
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026846, 0.0049872, 0.0027971, 0.0050781, -0.0016018, 0.0012931
1: 0.0017101, 0.0020428, 0.0017264, 0.0020559, -0.0002314, 0.0001868
2: 0.0116026, 0.0128756, 0.0115523, 0.0128134, -0.0007149, 0.0008856
3: -0.0026806, -0.0013639, -0.0027325, -0.0014282, -0.0007394, 0.0009159
4: -0.0025605, -0.0011351, -0.0024908, -0.0010788, -0.0009915, 0.0008005
5: 0.0051875, 0.0065364, 0.0051343, 0.0064705, -0.0007575, 0.0009383
6: -0.0017177, 0.0036342, -0.0019289, 0.0033727, -0.0030056, 0.0037230
7: -0.0075062, -0.0002174, -0.0071500, 0.0000703, -0.0050704, 0.0040934
8: 0.9839263, 0.9890608, 0.9841773, 0.9892634, -0.0035717, 0.0028835
9: -0.0059574, -0.0012967, -0.0061413, -0.0015244, -0.0026174, 0.0032422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017826, upper bound: 0.0017675
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017826, upper bound: 0.0017675
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027166, 0.0049629, 0.0026683, 0.0050340, -0.0014380, 0.0013241
1: 0.0017148, 0.0020393, 0.0017078, 0.0020496, -0.0002078, 0.0001913
2: 0.0116160, 0.0128579, 0.0115767, 0.0128846, -0.0007321, 0.0007951
3: -0.0026666, -0.0013822, -0.0027073, -0.0013545, -0.0007571, 0.0008223
4: -0.0025406, -0.0011502, -0.0025706, -0.0011061, -0.0008902, 0.0008196
5: 0.0052018, 0.0065176, 0.0051601, 0.0065460, -0.0007757, 0.0008424
6: -0.0016611, 0.0035598, -0.0018265, 0.0036722, -0.0030776, 0.0033424
7: -0.0074048, -0.0002945, -0.0075580, -0.0000692, -0.0045520, 0.0041914
8: 0.9839978, 0.9890064, 0.9838899, 0.9891652, -0.0032066, 0.0029525
9: -0.0059080, -0.0013615, -0.0060521, -0.0012636, -0.0026801, 0.0029107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0026846, 0.0049872, 0.0026683, 0.0050340, -0.0015541, 0.0014481
1: 0.0017101, 0.0020428, 0.0017078, 0.0020496, -0.0002245, 0.0002092
2: 0.0116026, 0.0128756, 0.0115767, 0.0128846, -0.0008006, 0.0008592
3: -0.0026806, -0.0013639, -0.0027073, -0.0013545, -0.0008281, 0.0008887
4: -0.0025605, -0.0011351, -0.0025706, -0.0011061, -0.0009620, 0.0008964
5: 0.0051875, 0.0065364, 0.0051601, 0.0065460, -0.0008483, 0.0009104
6: -0.0017177, 0.0036342, -0.0018265, 0.0036722, -0.0033659, 0.0036123
7: -0.0075062, -0.0002174, -0.0075580, -0.0000692, -0.0049196, 0.0045841
8: 0.9839263, 0.9890608, 0.9838899, 0.9891652, -0.0034655, 0.0032291
9: -0.0059574, -0.0012967, -0.0060521, -0.0012636, -0.0029312, 0.0031457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0025966, 0.0049114, 0.0027971, 0.0050781, -0.0016178, 0.0011721
1: 0.0016974, 0.0020319, 0.0017264, 0.0020559, -0.0002337, 0.0001693
2: 0.0116445, 0.0129243, 0.0115523, 0.0128134, -0.0006480, 0.0008944
3: -0.0026372, -0.0013136, -0.0027325, -0.0014282, -0.0006702, 0.0009251
4: -0.0026149, -0.0011820, -0.0024908, -0.0010788, -0.0010014, 0.0007256
5: 0.0052319, 0.0065880, 0.0051343, 0.0064705, -0.0006866, 0.0009477
6: -0.0015415, 0.0038388, -0.0019289, 0.0033727, -0.0027244, 0.0037601
7: -0.0077848, -0.0004573, -0.0071500, 0.0000703, -0.0051210, 0.0037104
8: 0.9837301, 0.9888917, 0.9841773, 0.9892634, -0.0036073, 0.0026137
9: -0.0058039, -0.0011185, -0.0061413, -0.0015244, -0.0023725, 0.0032745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017618
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017618
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025640, 0.0049310, 0.0027971, 0.0050781, -0.0017346, 0.0012812
1: 0.0016927, 0.0020347, 0.0017264, 0.0020559, -0.0002506, 0.0001851
2: 0.0116336, 0.0129423, 0.0115523, 0.0128134, -0.0007083, 0.0009590
3: -0.0026484, -0.0012949, -0.0027325, -0.0014282, -0.0007326, 0.0009918
4: -0.0026351, -0.0011699, -0.0024908, -0.0010788, -0.0010737, 0.0007931
5: 0.0052205, 0.0066070, 0.0051343, 0.0064705, -0.0007505, 0.0010161
6: -0.0015870, 0.0039145, -0.0019289, 0.0033727, -0.0029779, 0.0040316
7: -0.0078879, -0.0003954, -0.0071500, 0.0000703, -0.0054907, 0.0040556
8: 0.9836575, 0.9889354, 0.9841773, 0.9892634, -0.0038678, 0.0028569
9: -0.0058435, -0.0010526, -0.0061413, -0.0015244, -0.0025933, 0.0035109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017617
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017617
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0025966, 0.0049114, 0.0026680, 0.0050340, -0.0014638, 0.0011674
1: 0.0016974, 0.0020319, 0.0017077, 0.0020496, -0.0002115, 0.0001687
2: 0.0116445, 0.0129243, 0.0115767, 0.0128848, -0.0006454, 0.0008093
3: -0.0026372, -0.0013136, -0.0027073, -0.0013544, -0.0006675, 0.0008370
4: -0.0026149, -0.0011820, -0.0025708, -0.0011061, -0.0009061, 0.0007226
5: 0.0052319, 0.0065880, 0.0051601, 0.0065462, -0.0006839, 0.0008575
6: -0.0015415, 0.0038388, -0.0018265, 0.0036729, -0.0027133, 0.0034023
7: -0.0077848, -0.0004573, -0.0075589, -0.0000692, -0.0046336, 0.0036953
8: 0.9837301, 0.9888917, 0.9838892, 0.9891652, -0.0032640, 0.0026031
9: -0.0058039, -0.0011185, -0.0060521, -0.0012630, -0.0023629, 0.0029629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018421, upper bound: 0.0017849
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018421, upper bound: 0.0017849
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025640, 0.0049310, 0.0026680, 0.0050340, -0.0015847, 0.0012934
1: 0.0016927, 0.0020347, 0.0017077, 0.0020496, -0.0002289, 0.0001869
2: 0.0116336, 0.0129423, 0.0115767, 0.0128848, -0.0007151, 0.0008761
3: -0.0026484, -0.0012949, -0.0027073, -0.0013544, -0.0007396, 0.0009061
4: -0.0026351, -0.0011699, -0.0025708, -0.0011061, -0.0009809, 0.0008006
5: 0.0052205, 0.0066070, 0.0051601, 0.0065462, -0.0007577, 0.0009283
6: -0.0015870, 0.0039145, -0.0018265, 0.0036729, -0.0030063, 0.0036832
7: -0.0078879, -0.0003954, -0.0075589, -0.0000692, -0.0050162, 0.0040943
8: 0.9836575, 0.9889354, 0.9838892, 0.9891652, -0.0035335, 0.0028841
9: -0.0058435, -0.0010526, -0.0060521, -0.0012630, -0.0026180, 0.0032075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018421, upper bound: 0.0017848
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018421, upper bound: 0.0017848
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027382, 0.0049981, 0.0026208, 0.0049726, -0.0012279, 0.0012804
1: 0.0017179, 0.0020444, 0.0017009, 0.0020407, -0.0001774, 0.0001850
2: 0.0115965, 0.0128460, 0.0116107, 0.0129109, -0.0007079, 0.0006789
3: -0.0026868, -0.0013945, -0.0026722, -0.0013274, -0.0007321, 0.0007021
4: -0.0025273, -0.0011284, -0.0025999, -0.0011442, -0.0007601, 0.0007926
5: 0.0051812, 0.0065050, 0.0051961, 0.0065738, -0.0007500, 0.0007193
6: -0.0017430, 0.0035097, -0.0016836, 0.0037825, -0.0029760, 0.0028539
7: -0.0073365, -0.0001829, -0.0077081, -0.0002638, -0.0038868, 0.0040530
8: 0.9840458, 0.9890851, 0.9837841, 0.9890280, -0.0027380, 0.0028550
9: -0.0059794, -0.0014052, -0.0059277, -0.0011676, -0.0025916, 0.0024853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018149
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018646
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0027043, 0.0050226, 0.0026208, 0.0049726, -0.0011464, 0.0011809
1: 0.0017130, 0.0020479, 0.0017009, 0.0020407, -0.0001656, 0.0001706
2: 0.0115830, 0.0128647, 0.0116107, 0.0129109, -0.0006529, 0.0006338
3: -0.0027008, -0.0013751, -0.0026722, -0.0013274, -0.0006753, 0.0006555
4: -0.0025483, -0.0011132, -0.0025999, -0.0011442, -0.0007096, 0.0007310
5: 0.0051668, 0.0065249, 0.0051961, 0.0065738, -0.0006918, 0.0006716
6: -0.0017999, 0.0035885, -0.0016836, 0.0037825, -0.0027448, 0.0026646
7: -0.0074440, -0.0001054, -0.0077081, -0.0002638, -0.0036289, 0.0037381
8: 0.9839701, 0.9891396, 0.9837841, 0.9890280, -0.0025563, 0.0026332
9: -0.0060289, -0.0013365, -0.0059277, -0.0011676, -0.0023903, 0.0023204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018037
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018517
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027382, 0.0049981, 0.0026017, 0.0049302, -0.0012283, 0.0013683
1: 0.0017179, 0.0020444, 0.0016982, 0.0020346, -0.0001774, 0.0001977
2: 0.0115965, 0.0128460, 0.0116341, 0.0129214, -0.0007565, 0.0006791
3: -0.0026868, -0.0013945, -0.0026480, -0.0013165, -0.0007824, 0.0007023
4: -0.0025273, -0.0011284, -0.0026118, -0.0011704, -0.0007603, 0.0008470
5: 0.0051812, 0.0065050, 0.0052209, 0.0065850, -0.0008016, 0.0007195
6: -0.0017430, 0.0035097, -0.0015853, 0.0038269, -0.0031804, 0.0028548
7: -0.0073365, -0.0001829, -0.0077686, -0.0003977, -0.0038881, 0.0043314
8: 0.9840458, 0.9890851, 0.9837415, 0.9889337, -0.0027388, 0.0030511
9: -0.0059794, -0.0014052, -0.0058420, -0.0011289, -0.0027696, 0.0024861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018123
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018621
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0027043, 0.0050226, 0.0026017, 0.0049302, -0.0011450, 0.0012644
1: 0.0017130, 0.0020479, 0.0016982, 0.0020346, -0.0001654, 0.0001827
2: 0.0115830, 0.0128647, 0.0116341, 0.0129214, -0.0006990, 0.0006330
3: -0.0027008, -0.0013751, -0.0026480, -0.0013165, -0.0007230, 0.0006547
4: -0.0025483, -0.0011132, -0.0026118, -0.0011704, -0.0007088, 0.0007827
5: 0.0051668, 0.0065249, 0.0052209, 0.0065850, -0.0007407, 0.0006707
6: -0.0017999, 0.0035885, -0.0015853, 0.0038269, -0.0029387, 0.0026613
7: -0.0074440, -0.0001054, -0.0077686, -0.0003977, -0.0036245, 0.0040023
8: 0.9839701, 0.9891396, 0.9837415, 0.9889337, -0.0025531, 0.0028193
9: -0.0060289, -0.0013365, -0.0058420, -0.0011289, -0.0025592, 0.0023176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018008
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018493
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026168, 0.0049505, 0.0027010, 0.0050603, -0.0014090, 0.0012031
1: 0.0017003, 0.0020375, 0.0017125, 0.0020534, -0.0002036, 0.0001738
2: 0.0116229, 0.0129131, 0.0115622, 0.0128666, -0.0006651, 0.0007790
3: -0.0026596, -0.0013251, -0.0027223, -0.0013733, -0.0006879, 0.0008057
4: -0.0026025, -0.0011578, -0.0025503, -0.0010899, -0.0008722, 0.0007447
5: 0.0052091, 0.0065762, 0.0051447, 0.0065268, -0.0007047, 0.0008254
6: -0.0016323, 0.0037919, -0.0018875, 0.0035961, -0.0027962, 0.0032749
7: -0.0077210, -0.0003336, -0.0074543, 0.0000140, -0.0044602, 0.0038082
8: 0.9837750, 0.9889789, 0.9839629, 0.9892237, -0.0031418, 0.0026826
9: -0.0058830, -0.0011593, -0.0061053, -0.0013298, -0.0024351, 0.0028520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018059
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018059
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025818, 0.0049732, 0.0027010, 0.0050603, -0.0013088, 0.0010998
1: 0.0016953, 0.0020408, 0.0017125, 0.0020534, -0.0001891, 0.0001589
2: 0.0116103, 0.0129324, 0.0115622, 0.0128666, -0.0006081, 0.0007236
3: -0.0026725, -0.0013051, -0.0027223, -0.0013733, -0.0006289, 0.0007484
4: -0.0026241, -0.0011438, -0.0025503, -0.0010899, -0.0008101, 0.0006808
5: 0.0051957, 0.0065966, 0.0051447, 0.0065268, -0.0006443, 0.0007667
6: -0.0016851, 0.0038732, -0.0018875, 0.0035961, -0.0025563, 0.0030419
7: -0.0078316, -0.0002617, -0.0074543, 0.0000140, -0.0041428, 0.0034814
8: 0.9836971, 0.9890295, 0.9839629, 0.9892237, -0.0029183, 0.0024524
9: -0.0059290, -0.0010886, -0.0061053, -0.0013298, -0.0022261, 0.0026490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017934
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017934
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026168, 0.0049505, 0.0025763, 0.0050099, -0.0012285, 0.0011891
1: 0.0017003, 0.0020375, 0.0016945, 0.0020461, -0.0001775, 0.0001718
2: 0.0116229, 0.0129131, 0.0115900, 0.0129355, -0.0006574, 0.0006792
3: -0.0026596, -0.0013251, -0.0026935, -0.0013020, -0.0006799, 0.0007024
4: -0.0026025, -0.0011578, -0.0026275, -0.0011211, -0.0007604, 0.0007360
5: 0.0052091, 0.0065762, 0.0051743, 0.0065998, -0.0006966, 0.0007196
6: -0.0016323, 0.0037919, -0.0017704, 0.0038858, -0.0027637, 0.0028553
7: -0.0077210, -0.0003336, -0.0078489, -0.0001455, -0.0038886, 0.0037639
8: 0.9837750, 0.9889789, 0.9836849, 0.9891114, -0.0027392, 0.0026514
9: -0.0058830, -0.0011593, -0.0060033, -0.0010775, -0.0024068, 0.0024865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018264
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018264
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025818, 0.0049732, 0.0025763, 0.0050099, -0.0011265, 0.0010842
1: 0.0016953, 0.0020408, 0.0016945, 0.0020461, -0.0001627, 0.0001566
2: 0.0116103, 0.0129324, 0.0115900, 0.0129355, -0.0005994, 0.0006228
3: -0.0026725, -0.0013051, -0.0026935, -0.0013020, -0.0006200, 0.0006441
4: -0.0026241, -0.0011438, -0.0026275, -0.0011211, -0.0006973, 0.0006712
5: 0.0051957, 0.0065966, 0.0051743, 0.0065998, -0.0006351, 0.0006599
6: -0.0016851, 0.0038732, -0.0017704, 0.0038858, -0.0025200, 0.0026182
7: -0.0078316, -0.0002617, -0.0078489, -0.0001455, -0.0035657, 0.0034321
8: 0.9836971, 0.9890295, 0.9836849, 0.9891114, -0.0025118, 0.0024176
9: -0.0059290, -0.0010886, -0.0060033, -0.0010775, -0.0021946, 0.0022800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018154
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018154
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027382, 0.0049981, 0.0026735, 0.0050523, -0.0014497, 0.0013526
1: 0.0017179, 0.0020444, 0.0017085, 0.0020522, -0.0002094, 0.0001954
2: 0.0115965, 0.0128460, 0.0115666, 0.0128817, -0.0007478, 0.0008015
3: -0.0026868, -0.0013945, -0.0027178, -0.0013576, -0.0007734, 0.0008290
4: -0.0025273, -0.0011284, -0.0025673, -0.0010948, -0.0008974, 0.0008373
5: 0.0051812, 0.0065050, 0.0051494, 0.0065429, -0.0007924, 0.0008492
6: -0.0017430, 0.0035097, -0.0018690, 0.0036600, -0.0031439, 0.0033695
7: -0.0073365, -0.0001829, -0.0075413, -0.0000112, -0.0045890, 0.0042817
8: 0.9840458, 0.9890851, 0.9839016, 0.9892060, -0.0032326, 0.0030161
9: -0.0059794, -0.0014052, -0.0060892, -0.0012742, -0.0027378, 0.0029344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018036
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018563
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0027043, 0.0050226, 0.0026735, 0.0050523, -0.0013745, 0.0012787
1: 0.0017130, 0.0020479, 0.0017085, 0.0020522, -0.0001986, 0.0001847
2: 0.0115830, 0.0128647, 0.0115666, 0.0128817, -0.0007070, 0.0007599
3: -0.0027008, -0.0013751, -0.0027178, -0.0013576, -0.0007312, 0.0007859
4: -0.0025483, -0.0011132, -0.0025673, -0.0010948, -0.0008508, 0.0007915
5: 0.0051668, 0.0065249, 0.0051494, 0.0065429, -0.0007491, 0.0008052
6: -0.0017999, 0.0035885, -0.0018690, 0.0036600, -0.0029720, 0.0031946
7: -0.0074440, -0.0001054, -0.0075413, -0.0000112, -0.0043508, 0.0040477
8: 0.9839701, 0.9891396, 0.9839016, 0.9892060, -0.0030648, 0.0028513
9: -0.0060289, -0.0013365, -0.0060892, -0.0012742, -0.0025882, 0.0027820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0017885
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018463
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0027382, 0.0049981, 0.0026581, 0.0050153, -0.0014273, 0.0014124
1: 0.0017179, 0.0020444, 0.0017063, 0.0020469, -0.0002062, 0.0002040
2: 0.0115965, 0.0128460, 0.0115870, 0.0128903, -0.0007809, 0.0007891
3: -0.0026868, -0.0013945, -0.0026966, -0.0013487, -0.0008076, 0.0008161
4: -0.0025273, -0.0011284, -0.0025769, -0.0011177, -0.0008835, 0.0008743
5: 0.0051812, 0.0065050, 0.0051711, 0.0065520, -0.0008274, 0.0008361
6: -0.0017430, 0.0035097, -0.0017830, 0.0036959, -0.0032828, 0.0033175
7: -0.0073365, -0.0001829, -0.0075902, -0.0001284, -0.0045181, 0.0044709
8: 0.9840458, 0.9890851, 0.9838672, 0.9891235, -0.0031826, 0.0031494
9: -0.0059794, -0.0014052, -0.0060143, -0.0012430, -0.0028588, 0.0028890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018015
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018544
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0027043, 0.0050226, 0.0026581, 0.0050153, -0.0013474, 0.0013353
1: 0.0017130, 0.0020479, 0.0017063, 0.0020469, -0.0001947, 0.0001929
2: 0.0115830, 0.0128647, 0.0115870, 0.0128903, -0.0007383, 0.0007449
3: -0.0027008, -0.0013751, -0.0026966, -0.0013487, -0.0007635, 0.0007705
4: -0.0025483, -0.0011132, -0.0025769, -0.0011177, -0.0008341, 0.0008266
5: 0.0051668, 0.0065249, 0.0051711, 0.0065520, -0.0007822, 0.0007893
6: -0.0017999, 0.0035885, -0.0017830, 0.0036959, -0.0031036, 0.0031317
7: -0.0074440, -0.0001054, -0.0075902, -0.0001284, -0.0042651, 0.0042268
8: 0.9839701, 0.9891396, 0.9838672, 0.9891235, -0.0030045, 0.0029775
9: -0.0060289, -0.0013365, -0.0060143, -0.0012430, -0.0027027, 0.0027272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017862
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018445
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0026168, 0.0049505, 0.0027530, 0.0051409, -0.0016580, 0.0012734
1: 0.0017003, 0.0020375, 0.0017200, 0.0020650, -0.0002395, 0.0001840
2: 0.0116229, 0.0129131, 0.0115176, 0.0128378, -0.0007040, 0.0009167
3: -0.0026596, -0.0013251, -0.0027684, -0.0014030, -0.0007282, 0.0009480
4: -0.0026025, -0.0011578, -0.0025181, -0.0010400, -0.0010263, 0.0007883
5: 0.0052091, 0.0065762, 0.0050975, 0.0064963, -0.0007460, 0.0009712
6: -0.0016323, 0.0037919, -0.0020748, 0.0034752, -0.0029598, 0.0038536
7: -0.0077210, -0.0003336, -0.0072896, 0.0002690, -0.0052483, 0.0040310
8: 0.9837750, 0.9889789, 0.9840789, 0.9894034, -0.0036970, 0.0028395
9: -0.0058830, -0.0011593, -0.0062684, -0.0014352, -0.0025775, 0.0033559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017919
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017919
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0025818, 0.0049732, 0.0027530, 0.0051409, -0.0015868, 0.0012013
1: 0.0016953, 0.0020408, 0.0017200, 0.0020650, -0.0002293, 0.0001736
2: 0.0116103, 0.0129324, 0.0115176, 0.0128378, -0.0006642, 0.0008773
3: -0.0026725, -0.0013051, -0.0027684, -0.0014030, -0.0006869, 0.0009074
4: -0.0026241, -0.0011438, -0.0025181, -0.0010400, -0.0009823, 0.0007437
5: 0.0051957, 0.0065966, 0.0050975, 0.0064963, -0.0007037, 0.0009296
6: -0.0016851, 0.0038732, -0.0020748, 0.0034752, -0.0027923, 0.0036882
7: -0.0078316, -0.0002617, -0.0072896, 0.0002690, -0.0050231, 0.0038028
8: 0.9836971, 0.9890295, 0.9840789, 0.9894034, -0.0035384, 0.0026788
9: -0.0059290, -0.0010886, -0.0062684, -0.0014352, -0.0024316, 0.0032119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017766
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017766
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0026168, 0.0049505, 0.0026316, 0.0050911, -0.0014762, 0.0012633
1: 0.0017003, 0.0020375, 0.0017025, 0.0020578, -0.0002133, 0.0001825
2: 0.0116229, 0.0129131, 0.0115451, 0.0129049, -0.0006984, 0.0008161
3: -0.0026596, -0.0013251, -0.0027400, -0.0013336, -0.0007224, 0.0008441
4: -0.0026025, -0.0011578, -0.0025933, -0.0010708, -0.0009138, 0.0007820
5: 0.0052091, 0.0065762, 0.0051267, 0.0065675, -0.0007400, 0.0008647
6: -0.0016323, 0.0037919, -0.0019592, 0.0037575, -0.0029362, 0.0034311
7: -0.0077210, -0.0003336, -0.0076741, 0.0001116, -0.0046728, 0.0039989
8: 0.9837750, 0.9889789, 0.9838080, 0.9892925, -0.0032916, 0.0028169
9: -0.0058830, -0.0011593, -0.0061677, -0.0011893, -0.0025570, 0.0029879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0018146
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0018146
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0025818, 0.0049732, 0.0026316, 0.0050911, -0.0014050, 0.0011909
1: 0.0016953, 0.0020408, 0.0017025, 0.0020578, -0.0002030, 0.0001720
2: 0.0116103, 0.0129324, 0.0115451, 0.0129049, -0.0006584, 0.0007768
3: -0.0026725, -0.0013051, -0.0027400, -0.0013336, -0.0006810, 0.0008034
4: -0.0026241, -0.0011438, -0.0025933, -0.0010708, -0.0008697, 0.0007372
5: 0.0051957, 0.0065966, 0.0051267, 0.0065675, -0.0006976, 0.0008231
6: -0.0016851, 0.0038732, -0.0019592, 0.0037575, -0.0027679, 0.0032657
7: -0.0078316, -0.0002617, -0.0076741, 0.0001116, -0.0044475, 0.0037697
8: 0.9836971, 0.9890295, 0.9838080, 0.9892925, -0.0031329, 0.0026554
9: -0.0059290, -0.0010886, -0.0061677, -0.0011893, -0.0024104, 0.0028439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017999
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017999
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0027166, 0.0049629, 0.0027022, 0.0050324, -0.0013498, 0.0011905
1: 0.0017148, 0.0020393, 0.0017127, 0.0020493, -0.0001950, 0.0001720
2: 0.0116160, 0.0128579, 0.0115776, 0.0128659, -0.0006582, 0.0007463
3: -0.0026666, -0.0013822, -0.0027064, -0.0013739, -0.0006807, 0.0007718
4: -0.0025406, -0.0011502, -0.0025496, -0.0011071, -0.0008355, 0.0007369
5: 0.0052018, 0.0065176, 0.0051611, 0.0065261, -0.0006974, 0.0007907
6: -0.0016611, 0.0035598, -0.0018227, 0.0035934, -0.0027670, 0.0031373
7: -0.0074048, -0.0002945, -0.0074506, -0.0000743, -0.0042727, 0.0037684
8: 0.9839978, 0.9890064, 0.9839656, 0.9891616, -0.0030098, 0.0026546
9: -0.0059080, -0.0013615, -0.0060488, -0.0013323, -0.0024096, 0.0027321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017979
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017979
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0026846, 0.0049872, 0.0027022, 0.0050324, -0.0012502, 0.0010868
1: 0.0017101, 0.0020428, 0.0017127, 0.0020493, -0.0001806, 0.0001570
2: 0.0116026, 0.0128756, 0.0115776, 0.0128659, -0.0006009, 0.0006912
3: -0.0026806, -0.0013639, -0.0027064, -0.0013739, -0.0006214, 0.0007149
4: -0.0025605, -0.0011351, -0.0025496, -0.0011071, -0.0007739, 0.0006728
5: 0.0051875, 0.0065364, 0.0051611, 0.0065261, -0.0006367, 0.0007323
6: -0.0017177, 0.0036342, -0.0018227, 0.0035934, -0.0025260, 0.0029057
7: -0.0075062, -0.0002174, -0.0074506, -0.0000743, -0.0039573, 0.0034403
8: 0.9839263, 0.9890608, 0.9839656, 0.9891616, -0.0027876, 0.0024234
9: -0.0059574, -0.0012967, -0.0060488, -0.0013323, -0.0021998, 0.0025304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017858
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017858
time: 1.15 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.94 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018053
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018521
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018053
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018521
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018020
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018020
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018495
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017954
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018149
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018149
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018149
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018149
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0017860
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018437
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0017860
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018437
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017835
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017835
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018416
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017742
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017968
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017968
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017968
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017968
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017872
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017872
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017872
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017872
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018287
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018287
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018287
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018287
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017822
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018439, upper bound: 0.0018023
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018439, upper bound: 0.0018023
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018439, upper bound: 0.0018023
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018439, upper bound: 0.0018023
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017826, upper bound: 0.0017675
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017826, upper bound: 0.0017675
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017826, upper bound: 0.0017675
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017826, upper bound: 0.0017675
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018269
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017618
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017618
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017617
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017617
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018421, upper bound: 0.0017849
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018421, upper bound: 0.0017849
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018421, upper bound: 0.0017848
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018421, upper bound: 0.0017848
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018149
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018646
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018037
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017961, upper bound: 0.0018517
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018123
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018621
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018008
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018493
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018059
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0018059
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017934
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0017934
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018264
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018264
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018154
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018154
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018036
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018563
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0017885
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017935, upper bound: 0.0018463
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018015
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018544
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017862
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018445
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017919
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017919
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017766
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018251, upper bound: 0.0017766
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0018146
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0018146
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017999
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0018413, upper bound: 0.0017999
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017979
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017979
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017858
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.94
Output dim: 8, lower bound: -0.0017852, upper bound: 0.0017858
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018395
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017806, upper bound: 0.0018288
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017948
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0017808
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018156
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018282, upper bound: 0.0018035
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017854
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0017700
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018362
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017777, upper bound: 0.0018286
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017799
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017639
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0018030
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018260, upper bound: 0.0017877
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017706
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018332
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017921
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017855, upper bound: 0.0017581
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018152
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017804
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017803
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017885
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017723
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018486
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018351
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018486
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018351
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018102
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017949
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0018102
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018432, upper bound: 0.0017949
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017758
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017675, upper bound: 0.0017759
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017864, upper bound: 0.0017600
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018273
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018173
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018273
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0017614, upper bound: 0.0018173
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017981
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017831
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017981
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 8, lower bound: -0.0018434, upper bound: 0.0017831

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.07 + 597.22 = 600.29 seconds
