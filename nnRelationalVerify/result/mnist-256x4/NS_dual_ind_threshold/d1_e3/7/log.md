## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00056538


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0073481, 0.0081222, 0.0073481, 0.0081222, -0.0005468, 0.0005468)
1: (0.0023146, 0.0038140, 0.0023146, 0.0038140, -0.0010591, 0.0010591)
2: (-0.0138354, -0.0017417, -0.0138354, -0.0017417, -0.0085423, 0.0085423)
3: (-0.0023570, -0.0012769, -0.0023570, -0.0012769, -0.0007630, 0.0007630)
4: (0.0099004, 0.0151411, 0.0099004, 0.0151411, -0.0037017, 0.0037017)
5: (-0.0027800, -0.0019977, -0.0027800, -0.0019977, -0.0005526, 0.0005526)
6: (0.9939170, 0.9953517, 0.9939170, 0.9953517, -0.0010135, 0.0010135)
7: (0.0045385, 0.0140251, 0.0045385, 0.0140251, -0.0067008, 0.0067008)
8: (0.0024102, 0.0053823, 0.0024102, 0.0053823, -0.0020993, 0.0020993)
9: (-0.0180715, -0.0121396, -0.0180715, -0.0121396, -0.0041899, 0.0041899)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 2.30 = 3.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0006282, upper bound: 0.0006282

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006091, upper bound: 0.0005775
time: 1.39 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006091, upper bound: 0.0006091
time: 1.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.88 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.88
Output dim: 6, lower bound: -0.0006091, upper bound: 0.0005775
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.88
Output dim: 6, lower bound: -0.0006091, upper bound: 0.0006091

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0073722, 0.0080826, 0.0073489, 0.0081099, -0.0005123, 0.0005097
1: 0.0023613, 0.0037373, 0.0023163, 0.0037901, -0.0009924, 0.0009872
2: -0.0132171, -0.0021183, -0.0136430, -0.0017555, -0.0079629, 0.0080042
3: -0.0023234, -0.0013321, -0.0023558, -0.0012940, -0.0007149, 0.0007112
4: 0.0100636, 0.0148731, 0.0099063, 0.0150577, -0.0034686, 0.0034506
5: -0.0027400, -0.0020220, -0.0027675, -0.0019985, -0.0005151, 0.0005178
6: 0.9939616, 0.9952784, 0.9939186, 0.9953289, -0.0009497, 0.0009447
7: 0.0048339, 0.0135400, 0.0045493, 0.0138741, -0.0062787, 0.0062463
8: 0.0025028, 0.0052303, 0.0024136, 0.0053350, -0.0019671, 0.0019569
9: -0.0177682, -0.0123244, -0.0179771, -0.0121464, -0.0039057, 0.0039260

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005713, upper bound: 0.0005612
time: 1.38 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005938, upper bound: 0.0005632
time: 1.45 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0073506, 0.0081053, 0.0073488, 0.0081170, -0.0005392, 0.0005042
1: 0.0023195, 0.0037813, 0.0023161, 0.0038040, -0.0010444, 0.0009766
2: -0.0135715, -0.0017811, -0.0137550, -0.0017534, -0.0078768, 0.0084238
3: -0.0023535, -0.0013004, -0.0023560, -0.0012840, -0.0007524, 0.0007035
4: 0.0099175, 0.0150267, 0.0099054, 0.0151062, -0.0036504, 0.0034133
5: -0.0027629, -0.0020002, -0.0027748, -0.0019984, -0.0005095, 0.0005449
6: 0.9939216, 0.9953204, 0.9939183, 0.9953422, -0.0009994, 0.0009345
7: 0.0045695, 0.0138180, 0.0045477, 0.0139620, -0.0066078, 0.0061787
8: 0.0024199, 0.0053174, 0.0024131, 0.0053625, -0.0020702, 0.0019358
9: -0.0179420, -0.0121590, -0.0180320, -0.0121454, -0.0038635, 0.0041318

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005712, upper bound: 0.0005918
time: 1.35 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005938, upper bound: 0.0005938
time: 1.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.96 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 6, lower bound: -0.0005713, upper bound: 0.0005612
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 6, lower bound: -0.0005938, upper bound: 0.0005632
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 6, lower bound: -0.0005712, upper bound: 0.0005918
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 6, lower bound: -0.0005938, upper bound: 0.0005938

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0073733, 0.0080703, 0.0073527, 0.0080737, -0.0004771, 0.0004964
1: 0.0023635, 0.0037135, 0.0023236, 0.0037201, -0.0009241, 0.0009615
2: -0.0130245, -0.0021361, -0.0130783, -0.0018140, -0.0077554, 0.0074539
3: -0.0023218, -0.0013493, -0.0023506, -0.0013445, -0.0006657, 0.0006927
4: 0.0100713, 0.0147896, 0.0099317, 0.0148130, -0.0032301, 0.0033607
5: -0.0027275, -0.0020232, -0.0027310, -0.0020023, -0.0005017, 0.0004822
6: 0.9939637, 0.9952556, 0.9939256, 0.9952620, -0.0008844, 0.0009201
7: 0.0048479, 0.0133890, 0.0045953, 0.0134312, -0.0058470, 0.0060835
8: 0.0025072, 0.0051830, 0.0024280, 0.0051962, -0.0018318, 0.0019059
9: -0.0176737, -0.0123331, -0.0177001, -0.0121751, -0.0038039, 0.0036561

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005562, upper bound: 0.0005296
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005572, upper bound: 0.0005474
time: 1.36 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0073726, 0.0080801, 0.0073508, 0.0080987, -0.0004828, 0.0005060
1: 0.0023621, 0.0037326, 0.0023199, 0.0037685, -0.0009350, 0.0009801
2: -0.0131786, -0.0021244, -0.0134682, -0.0017844, -0.0079055, 0.0075420
3: -0.0023228, -0.0013355, -0.0023532, -0.0013097, -0.0006736, 0.0007061
4: 0.0100662, 0.0148564, 0.0099189, 0.0149819, -0.0032682, 0.0034258
5: -0.0027375, -0.0020224, -0.0027562, -0.0020004, -0.0005114, 0.0004879
6: 0.9939624, 0.9952739, 0.9939220, 0.9953082, -0.0008948, 0.0009379
7: 0.0048388, 0.0135098, 0.0045720, 0.0137371, -0.0059160, 0.0062012
8: 0.0025043, 0.0052209, 0.0024207, 0.0052921, -0.0018535, 0.0019428
9: -0.0177493, -0.0123274, -0.0178914, -0.0121606, -0.0038776, 0.0036993

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005799, upper bound: 0.0005301
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005817, upper bound: 0.0005497
time: 1.51 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0073517, 0.0080932, 0.0073526, 0.0080812, -0.0005051, 0.0004862
1: 0.0023217, 0.0037579, 0.0023233, 0.0037347, -0.0009783, 0.0009417
2: -0.0133827, -0.0017990, -0.0131955, -0.0018118, -0.0075955, 0.0078912
3: -0.0023519, -0.0013173, -0.0023508, -0.0013340, -0.0007048, 0.0006784
4: 0.0099252, 0.0149449, 0.0099307, 0.0148638, -0.0034196, 0.0032914
5: -0.0027507, -0.0020014, -0.0027386, -0.0020022, -0.0004913, 0.0005105
6: 0.9939237, 0.9952980, 0.9939253, 0.9952759, -0.0009362, 0.0009012
7: 0.0045835, 0.0136699, 0.0045935, 0.0135231, -0.0061900, 0.0059580
8: 0.0024243, 0.0052710, 0.0024275, 0.0052250, -0.0019393, 0.0018666
9: -0.0178494, -0.0121678, -0.0177576, -0.0121740, -0.0037255, 0.0038706

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005562, upper bound: 0.0005513
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005573, upper bound: 0.0005796
time: 1.50 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073510, 0.0081027, 0.0073507, 0.0081057, -0.0005151, 0.0005006
1: 0.0023203, 0.0037763, 0.0023197, 0.0037820, -0.0009976, 0.0009695
2: -0.0135313, -0.0017874, -0.0135775, -0.0017822, -0.0078202, 0.0080466
3: -0.0023529, -0.0013040, -0.0023534, -0.0012999, -0.0007187, 0.0006985
4: 0.0099202, 0.0150093, 0.0099179, 0.0150293, -0.0034869, 0.0033888
5: -0.0027603, -0.0020006, -0.0027633, -0.0020003, -0.0005059, 0.0005205
6: 0.9939224, 0.9953157, 0.9939218, 0.9953212, -0.0009547, 0.0009278
7: 0.0045744, 0.0137865, 0.0045703, 0.0138227, -0.0063119, 0.0061343
8: 0.0024215, 0.0053076, 0.0024202, 0.0053189, -0.0019775, 0.0019218
9: -0.0179223, -0.0121621, -0.0179450, -0.0121595, -0.0038357, 0.0039468

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005799, upper bound: 0.0005522
time: 1.42 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005818, upper bound: 0.0005818
time: 1.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.21 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.21
Output dim: 6, lower bound: -0.0005562, upper bound: 0.0005296
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.21
Output dim: 6, lower bound: -0.0005572, upper bound: 0.0005474
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 6, lower bound: -0.0005799, upper bound: 0.0005301
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 6, lower bound: -0.0005817, upper bound: 0.0005497
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.21
Output dim: 6, lower bound: -0.0005562, upper bound: 0.0005513
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 6, lower bound: -0.0005573, upper bound: 0.0005796
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 6, lower bound: -0.0005799, upper bound: 0.0005522
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.21
Output dim: 6, lower bound: -0.0005818, upper bound: 0.0005818

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0073759, 0.0080453, 0.0073524, 0.0080871, -0.0004704, 0.0004699
1: 0.0023685, 0.0036651, 0.0023230, 0.0037461, -0.0009110, 0.0009101
2: -0.0126341, -0.0021762, -0.0132876, -0.0018089, -0.0073410, 0.0073484
3: -0.0023182, -0.0013842, -0.0023510, -0.0013258, -0.0006563, 0.0006557
4: 0.0100886, 0.0146205, 0.0099295, 0.0149037, -0.0031843, 0.0031811
5: -0.0027023, -0.0020258, -0.0027445, -0.0020020, -0.0004749, 0.0004754
6: 0.9939685, 0.9952093, 0.9939249, 0.9952868, -0.0008718, 0.0008710
7: 0.0048793, 0.0130828, 0.0045913, 0.0135954, -0.0057642, 0.0057584
8: 0.0025170, 0.0050871, 0.0024268, 0.0052477, -0.0018059, 0.0018041
9: -0.0174823, -0.0123527, -0.0178028, -0.0121726, -0.0036007, 0.0036043

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005601, upper bound: 0.0005301
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005601, upper bound: 0.0005301
time: 2.11 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0073745, 0.0080719, 0.0073510, 0.0080977, -0.0004778, 0.0004789
1: 0.0023659, 0.0037167, 0.0023203, 0.0037666, -0.0009254, 0.0009277
2: -0.0130503, -0.0021553, -0.0134530, -0.0017878, -0.0074825, 0.0074645
3: -0.0023201, -0.0013470, -0.0023529, -0.0013110, -0.0006667, 0.0006683
4: 0.0100796, 0.0148008, 0.0099203, 0.0149753, -0.0032347, 0.0032425
5: -0.0027292, -0.0020244, -0.0027552, -0.0020006, -0.0004840, 0.0004829
6: 0.9939660, 0.9952587, 0.9939224, 0.9953063, -0.0008856, 0.0008878
7: 0.0048630, 0.0134092, 0.0045747, 0.0137251, -0.0058553, 0.0058694
8: 0.0025119, 0.0051893, 0.0024216, 0.0052883, -0.0018344, 0.0018389
9: -0.0176864, -0.0123425, -0.0178839, -0.0121623, -0.0036701, 0.0036613

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005623, upper bound: 0.0005497
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005623, upper bound: 0.0005496
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0073536, 0.0080856, 0.0073528, 0.0080802, -0.0005002, 0.0004677
1: 0.0023254, 0.0037431, 0.0023237, 0.0037328, -0.0009689, 0.0009059
2: -0.0132638, -0.0018285, -0.0131802, -0.0018151, -0.0073066, 0.0078149
3: -0.0023493, -0.0013279, -0.0023505, -0.0013354, -0.0006980, 0.0006526
4: 0.0099380, 0.0148934, 0.0099322, 0.0148571, -0.0033865, 0.0031662
5: -0.0027430, -0.0020033, -0.0027376, -0.0020024, -0.0004727, 0.0005055
6: 0.9939272, 0.9952839, 0.9939256, 0.9952740, -0.0009272, 0.0008669
7: 0.0046066, 0.0135767, 0.0045961, 0.0135112, -0.0061302, 0.0057314
8: 0.0024316, 0.0052418, 0.0024283, 0.0052213, -0.0019205, 0.0017956
9: -0.0177911, -0.0121822, -0.0177501, -0.0121756, -0.0035838, 0.0038331

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0073558, 0.0080668, 0.0073522, 0.0080940, -0.0005032, 0.0004641
1: 0.0023295, 0.0037068, 0.0023227, 0.0037594, -0.0009746, 0.0008989
2: -0.0129704, -0.0018617, -0.0133948, -0.0018068, -0.0072508, 0.0078609
3: -0.0023463, -0.0013541, -0.0023512, -0.0013162, -0.0007021, 0.0006476
4: 0.0099524, 0.0147662, 0.0099286, 0.0149501, -0.0034064, 0.0031421
5: -0.0027240, -0.0020054, -0.0027515, -0.0020019, -0.0004690, 0.0005085
6: 0.9939311, 0.9952492, 0.9939246, 0.9952995, -0.0009326, 0.0008603
7: 0.0046327, 0.0133466, 0.0045896, 0.0136795, -0.0061662, 0.0056877
8: 0.0024397, 0.0051697, 0.0024262, 0.0052740, -0.0019318, 0.0017819
9: -0.0176472, -0.0121985, -0.0178554, -0.0121716, -0.0035565, 0.0038557

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005474, upper bound: 0.0005523
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005474, upper bound: 0.0005522
time: 1.47 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0073529, 0.0080951, 0.0073509, 0.0081047, -0.0005103, 0.0004747
1: 0.0023240, 0.0037616, 0.0023201, 0.0037802, -0.0009884, 0.0009194
2: -0.0134126, -0.0018170, -0.0135625, -0.0017857, -0.0074156, 0.0079723
3: -0.0023503, -0.0013146, -0.0023531, -0.0013012, -0.0007121, 0.0006623
4: 0.0099330, 0.0149578, 0.0099194, 0.0150228, -0.0034547, 0.0032135
5: -0.0027526, -0.0020025, -0.0027623, -0.0020005, -0.0004797, 0.0005157
6: 0.9939258, 0.9953016, 0.9939222, 0.9953194, -0.0009459, 0.0008798
7: 0.0045976, 0.0136934, 0.0045730, 0.0138110, -0.0062537, 0.0058169
8: 0.0024287, 0.0052784, 0.0024210, 0.0053152, -0.0019592, 0.0018224
9: -0.0178641, -0.0121766, -0.0179376, -0.0121612, -0.0036373, 0.0039104

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005817
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005818
time: 1.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.79 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005601, upper bound: 0.0005301
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005601, upper bound: 0.0005301
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005623, upper bound: 0.0005497
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005623, upper bound: 0.0005496
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005474, upper bound: 0.0005523
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005474, upper bound: 0.0005522
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005817
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.79
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005818

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0073536, 0.0080856, 0.0073776, 0.0080455, -0.0004670, 0.0004726
1: 0.0023254, 0.0037431, 0.0023718, 0.0036654, -0.0009045, 0.0009154
2: -0.0132638, -0.0018285, -0.0126369, -0.0022028, -0.0073838, 0.0072954
3: -0.0023493, -0.0013279, -0.0023158, -0.0013839, -0.0006516, 0.0006595
4: 0.0099380, 0.0148934, 0.0101002, 0.0146217, -0.0031614, 0.0031997
5: -0.0027430, -0.0020033, -0.0027024, -0.0020275, -0.0004776, 0.0004719
6: 0.9939272, 0.9952839, 0.9939716, 0.9952096, -0.0008655, 0.0008760
7: 0.0046066, 0.0135767, 0.0049002, 0.0130849, -0.0057226, 0.0057920
8: 0.0024316, 0.0052418, 0.0025236, 0.0050878, -0.0017929, 0.0018146
9: -0.0177911, -0.0121822, -0.0174836, -0.0123658, -0.0036217, 0.0035783

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005640
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073536, 0.0080856, 0.0073546, 0.0080690, -0.0004611, 0.0004655
1: 0.0023254, 0.0037431, 0.0023273, 0.0037111, -0.0008932, 0.0009016
2: -0.0132638, -0.0018285, -0.0130053, -0.0018435, -0.0072721, 0.0072041
3: -0.0023493, -0.0013279, -0.0023479, -0.0013510, -0.0006434, 0.0006495
4: 0.0099380, 0.0148934, 0.0099445, 0.0147813, -0.0031218, 0.0031513
5: -0.0027430, -0.0020033, -0.0027263, -0.0020042, -0.0004704, 0.0004660
6: 0.9939272, 0.9952839, 0.9939290, 0.9952533, -0.0008547, 0.0008628
7: 0.0046066, 0.0135767, 0.0046184, 0.0133739, -0.0056511, 0.0057044
8: 0.0024316, 0.0052418, 0.0024353, 0.0051783, -0.0017704, 0.0017871
9: -0.0177911, -0.0121822, -0.0176643, -0.0121896, -0.0035669, 0.0035336

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005640
time: 2.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
time: 2.16 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0073529, 0.0080951, 0.0073742, 0.0080703, -0.0004777, 0.0004787
1: 0.0023240, 0.0037616, 0.0023653, 0.0037135, -0.0009252, 0.0009272
2: -0.0134126, -0.0018170, -0.0130247, -0.0021506, -0.0074785, 0.0074629
3: -0.0023503, -0.0013146, -0.0023205, -0.0013493, -0.0006665, 0.0006679
4: 0.0099330, 0.0149578, 0.0100776, 0.0147897, -0.0032340, 0.0032407
5: -0.0027526, -0.0020025, -0.0027275, -0.0020241, -0.0004838, 0.0004828
6: 0.9939258, 0.9953016, 0.9939655, 0.9952555, -0.0008854, 0.0008873
7: 0.0045976, 0.0136934, 0.0048593, 0.0133891, -0.0058540, 0.0058663
8: 0.0024287, 0.0052784, 0.0025107, 0.0051831, -0.0018340, 0.0018379
9: -0.0178641, -0.0121766, -0.0176738, -0.0123402, -0.0036681, 0.0036605

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005573
time: 1.96 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005818
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073529, 0.0080951, 0.0073527, 0.0080925, -0.0004663, 0.0004725
1: 0.0023240, 0.0037616, 0.0023235, 0.0037566, -0.0009033, 0.0009152
2: -0.0134126, -0.0018170, -0.0133724, -0.0018136, -0.0073822, 0.0072856
3: -0.0023503, -0.0013146, -0.0023506, -0.0013182, -0.0006507, 0.0006593
4: 0.0099330, 0.0149578, 0.0099315, 0.0149404, -0.0031571, 0.0031990
5: -0.0027526, -0.0020025, -0.0027500, -0.0020023, -0.0004775, 0.0004713
6: 0.9939258, 0.9953016, 0.9939255, 0.9952968, -0.0008644, 0.0008758
7: 0.0045976, 0.0136934, 0.0045949, 0.0136619, -0.0057150, 0.0057907
8: 0.0024287, 0.0052784, 0.0024279, 0.0052685, -0.0017905, 0.0018142
9: -0.0178641, -0.0121766, -0.0178444, -0.0121749, -0.0036209, 0.0035735

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005573
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005817
time: 1.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.16 seconds
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005640
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005640
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005573
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005818
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005573
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005817

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073776, 0.0080455, -0.0004664, 0.0004714
1: 0.0023267, 0.0037436, 0.0023718, 0.0036654, -0.0009033, 0.0009131
2: -0.0132678, -0.0018394, -0.0126369, -0.0022028, -0.0073646, 0.0072859
3: -0.0023483, -0.0013276, -0.0023158, -0.0013839, -0.0006507, 0.0006578
4: 0.0099427, 0.0148951, 0.0101002, 0.0146217, -0.0031573, 0.0031914
5: -0.0027433, -0.0020040, -0.0027024, -0.0020275, -0.0004764, 0.0004713
6: 0.9939286, 0.9952844, 0.9939716, 0.9952096, -0.0008644, 0.0008738
7: 0.0046151, 0.0135799, 0.0049002, 0.0130849, -0.0057152, 0.0057770
8: 0.0024342, 0.0052428, 0.0025236, 0.0050878, -0.0017905, 0.0018099
9: -0.0177931, -0.0121875, -0.0174836, -0.0123658, -0.0036123, 0.0035737

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005786
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005796
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073546, 0.0080690, -0.0004605, 0.0004677
1: 0.0023267, 0.0037436, 0.0023273, 0.0037111, -0.0008920, 0.0009059
2: -0.0132678, -0.0018394, -0.0130053, -0.0018435, -0.0073072, 0.0071950
3: -0.0023483, -0.0013276, -0.0023479, -0.0013510, -0.0006426, 0.0006526
4: 0.0099427, 0.0148951, 0.0099445, 0.0147813, -0.0031179, 0.0031665
5: -0.0027433, -0.0020040, -0.0027263, -0.0020042, -0.0004727, 0.0004654
6: 0.9939286, 0.9952844, 0.9939290, 0.9952533, -0.0008536, 0.0008670
7: 0.0046151, 0.0135799, 0.0046184, 0.0133739, -0.0056439, 0.0057319
8: 0.0024342, 0.0052428, 0.0024353, 0.0051783, -0.0017682, 0.0017958
9: -0.0177931, -0.0121875, -0.0176643, -0.0121896, -0.0035841, 0.0035291

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005786
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005796
time: 2.11 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073742, 0.0080703, -0.0004764, 0.0004608
1: 0.0023267, 0.0037436, 0.0023653, 0.0037135, -0.0009228, 0.0008926
2: -0.0132678, -0.0018394, -0.0130247, -0.0021506, -0.0071993, 0.0074429
3: -0.0023483, -0.0013276, -0.0023205, -0.0013493, -0.0006648, 0.0006430
4: 0.0099427, 0.0148951, 0.0100776, 0.0147897, -0.0032253, 0.0031198
5: -0.0027433, -0.0020040, -0.0027275, -0.0020241, -0.0004657, 0.0004815
6: 0.9939286, 0.9952844, 0.9939655, 0.9952555, -0.0008830, 0.0008542
7: 0.0046151, 0.0135799, 0.0048593, 0.0133891, -0.0058383, 0.0056473
8: 0.0024342, 0.0052428, 0.0025107, 0.0051831, -0.0018291, 0.0017693
9: -0.0177931, -0.0121875, -0.0176738, -0.0123402, -0.0035312, 0.0036506

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005799
time: 1.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005818
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073527, 0.0080925, -0.0004651, 0.0004518
1: 0.0023267, 0.0037436, 0.0023235, 0.0037566, -0.0009009, 0.0008751
2: -0.0132678, -0.0018394, -0.0133724, -0.0018136, -0.0070588, 0.0072664
3: -0.0023483, -0.0013276, -0.0023506, -0.0013182, -0.0006490, 0.0006305
4: 0.0099427, 0.0148951, 0.0099315, 0.0149404, -0.0031488, 0.0030589
5: -0.0027433, -0.0020040, -0.0027500, -0.0020023, -0.0004566, 0.0004701
6: 0.9939286, 0.9952844, 0.9939255, 0.9952968, -0.0008621, 0.0008375
7: 0.0046151, 0.0135799, 0.0045949, 0.0136619, -0.0056999, 0.0055371
8: 0.0024342, 0.0052428, 0.0024279, 0.0052685, -0.0017857, 0.0017347
9: -0.0177931, -0.0121875, -0.0178444, -0.0121749, -0.0034623, 0.0035641

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005799
time: 2.10 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005573
time: 1.49 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.96 seconds
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005786
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005796
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005786
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005796
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005799
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005818
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.96
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005799
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 6, lower bound: -0.0005134, upper bound: 0.0005573

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073790, 0.0080128, -0.0004370, 0.0004863
1: 0.0023267, 0.0037436, 0.0023746, 0.0036021, -0.0008464, 0.0009420
2: -0.0132678, -0.0018394, -0.0121263, -0.0022254, -0.0075981, 0.0068271
3: -0.0023483, -0.0013276, -0.0023138, -0.0014295, -0.0006098, 0.0006786
4: 0.0099427, 0.0148951, 0.0101100, 0.0144004, -0.0029584, 0.0032926
5: -0.0027433, -0.0020040, -0.0026694, -0.0020289, -0.0004915, 0.0004416
6: 0.9939286, 0.9952844, 0.9939743, 0.9951490, -0.0008100, 0.0009015
7: 0.0046151, 0.0135799, 0.0049180, 0.0126844, -0.0053553, 0.0059601
8: 0.0024342, 0.0052428, 0.0025291, 0.0049623, -0.0016778, 0.0018673
9: -0.0177931, -0.0121875, -0.0172332, -0.0123769, -0.0037268, 0.0033486

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005625
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005002, upper bound: 0.0005636
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073793, 0.0080381, -0.0004448, 0.0004685
1: 0.0023267, 0.0037436, 0.0023750, 0.0036512, -0.0008616, 0.0009073
2: -0.0132678, -0.0018394, -0.0125221, -0.0022289, -0.0073186, 0.0069492
3: -0.0023483, -0.0013276, -0.0023135, -0.0013942, -0.0006207, 0.0006537
4: 0.0099427, 0.0148951, 0.0101115, 0.0145719, -0.0030114, 0.0031714
5: -0.0027433, -0.0020040, -0.0026950, -0.0020292, -0.0004734, 0.0004495
6: 0.9939286, 0.9952844, 0.9939747, 0.9951959, -0.0008245, 0.0008683
7: 0.0046151, 0.0135799, 0.0049207, 0.0129949, -0.0054511, 0.0057408
8: 0.0024342, 0.0052428, 0.0025300, 0.0050595, -0.0017078, 0.0017986
9: -0.0177931, -0.0121875, -0.0174273, -0.0123786, -0.0035897, 0.0034085

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005636
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005647
time: 2.10 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073573, 0.0080353, -0.0004310, 0.0004814
1: 0.0023267, 0.0037436, 0.0023326, 0.0036458, -0.0008347, 0.0009324
2: -0.0132678, -0.0018394, -0.0124788, -0.0018863, -0.0075209, 0.0067328
3: -0.0023483, -0.0013276, -0.0023441, -0.0013980, -0.0006013, 0.0006717
4: 0.0099427, 0.0148951, 0.0099630, 0.0145532, -0.0029176, 0.0032591
5: -0.0027433, -0.0020040, -0.0026922, -0.0020070, -0.0004865, 0.0004355
6: 0.9939286, 0.9952844, 0.9939341, 0.9951908, -0.0007988, 0.0008923
7: 0.0046151, 0.0135799, 0.0046520, 0.0129609, -0.0052813, 0.0058995
8: 0.0024342, 0.0052428, 0.0024458, 0.0050489, -0.0016546, 0.0018483
9: -0.0177931, -0.0121875, -0.0174061, -0.0122106, -0.0036889, 0.0033024

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005625
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005635
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073562, 0.0080619, -0.0004402, 0.0004648
1: 0.0023267, 0.0037436, 0.0023304, 0.0036974, -0.0008527, 0.0009003
2: -0.0132678, -0.0018394, -0.0128945, -0.0018685, -0.0072620, 0.0068776
3: -0.0023483, -0.0013276, -0.0023457, -0.0013609, -0.0006143, 0.0006486
4: 0.0099427, 0.0148951, 0.0099553, 0.0147333, -0.0029804, 0.0031469
5: -0.0027433, -0.0020040, -0.0027191, -0.0020059, -0.0004698, 0.0004449
6: 0.9939286, 0.9952844, 0.9939319, 0.9952402, -0.0008160, 0.0008616
7: 0.0046151, 0.0135799, 0.0046380, 0.0132870, -0.0053950, 0.0056965
8: 0.0024342, 0.0052428, 0.0024414, 0.0051511, -0.0016902, 0.0017847
9: -0.0177931, -0.0121875, -0.0176100, -0.0122019, -0.0035619, 0.0033734

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005636
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005647
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073772, 0.0080364, -0.0004460, 0.0004711
1: 0.0023267, 0.0037436, 0.0023711, 0.0036478, -0.0008638, 0.0009126
2: -0.0132678, -0.0018394, -0.0124948, -0.0021973, -0.0073606, 0.0069674
3: -0.0023483, -0.0013276, -0.0023163, -0.0013966, -0.0006223, 0.0006574
4: 0.0099427, 0.0148951, 0.0100978, 0.0145601, -0.0030193, 0.0031896
5: -0.0027433, -0.0020040, -0.0026933, -0.0020271, -0.0004761, 0.0004507
6: 0.9939286, 0.9952844, 0.9939710, 0.9951927, -0.0008266, 0.0008733
7: 0.0046151, 0.0135799, 0.0048959, 0.0129735, -0.0054654, 0.0057738
8: 0.0024342, 0.0052428, 0.0025222, 0.0050529, -0.0017123, 0.0018089
9: -0.0177931, -0.0121875, -0.0174140, -0.0123631, -0.0036103, 0.0034175

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005639
time: 1.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005649
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073760, 0.0080631, -0.0004591, 0.0004578
1: 0.0023267, 0.0037436, 0.0023687, 0.0036997, -0.0008893, 0.0008868
2: -0.0132678, -0.0018394, -0.0129132, -0.0021775, -0.0071525, 0.0071729
3: -0.0023483, -0.0013276, -0.0023181, -0.0013592, -0.0006406, 0.0006388
4: 0.0099427, 0.0148951, 0.0100892, 0.0147414, -0.0031083, 0.0030995
5: -0.0027433, -0.0020040, -0.0027203, -0.0020258, -0.0004627, 0.0004640
6: 0.9939286, 0.9952844, 0.9939685, 0.9952424, -0.0008510, 0.0008486
7: 0.0046151, 0.0135799, 0.0048804, 0.0133017, -0.0056265, 0.0056106
8: 0.0024342, 0.0052428, 0.0025173, 0.0051557, -0.0017628, 0.0017578
9: -0.0177931, -0.0121875, -0.0176192, -0.0123534, -0.0035082, 0.0035182

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005660
time: 1.45 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005671
time: 1.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0073543, 0.0080858, 0.0073571, 0.0080575, -0.0004345, 0.0004607
1: 0.0023267, 0.0037436, 0.0023322, 0.0036888, -0.0008416, 0.0008924
2: -0.0132678, -0.0018394, -0.0128257, -0.0018833, -0.0071979, 0.0067882
3: -0.0023483, -0.0013276, -0.0023444, -0.0013670, -0.0006063, 0.0006429
4: 0.0099427, 0.0148951, 0.0099617, 0.0147035, -0.0029416, 0.0031191
5: -0.0027433, -0.0020040, -0.0027147, -0.0020068, -0.0004656, 0.0004391
6: 0.9939286, 0.9952844, 0.9939336, 0.9952319, -0.0008054, 0.0008540
7: 0.0046151, 0.0135799, 0.0046496, 0.0132330, -0.0053248, 0.0056461
8: 0.0024342, 0.0052428, 0.0024450, 0.0051342, -0.0016682, 0.0017689
9: -0.0177931, -0.0121875, -0.0175762, -0.0122091, -0.0035305, 0.0033296

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005639
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005649
time: 1.61 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.46 seconds
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005625
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005002, upper bound: 0.0005636
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005636
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005647
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005625
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005635
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005636
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005647
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005639
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005649
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005660
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005671
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005639
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.46
Output dim: 6, lower bound: -0.0005025, upper bound: 0.0005649

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0073687, 0.0080853, 0.0073803, 0.0080629, -0.0004422, 0.0004519
1: 0.0023546, 0.0037425, 0.0023770, 0.0036992, -0.0008565, 0.0008753
2: -0.0132588, -0.0020637, -0.0129092, -0.0022450, -0.0070598, 0.0069084
3: -0.0023282, -0.0013284, -0.0023121, -0.0013596, -0.0006170, 0.0006305
4: 0.0100399, 0.0148912, 0.0101185, 0.0147397, -0.0029937, 0.0030593
5: -0.0027427, -0.0020185, -0.0027201, -0.0020302, -0.0004567, 0.0004469
6: 0.9939552, 0.9952834, 0.9939766, 0.9952419, -0.0008196, 0.0008376
7: 0.0047911, 0.0135728, 0.0049333, 0.0132986, -0.0054191, 0.0055379
8: 0.0024894, 0.0052406, 0.0025339, 0.0051547, -0.0016978, 0.0017350
9: -0.0177887, -0.0122976, -0.0176172, -0.0123865, -0.0034628, 0.0033885

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004818, upper bound: 0.0005551
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004958, upper bound: 0.0005551
time: 1.84 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0073711, 0.0080823, 0.0073816, 0.0080630, -0.0004424, 0.0004567
1: 0.0023592, 0.0037368, 0.0023796, 0.0036993, -0.0008569, 0.0008845
2: -0.0132125, -0.0021010, -0.0129105, -0.0022655, -0.0071345, 0.0069113
3: -0.0023249, -0.0013325, -0.0023102, -0.0013595, -0.0006173, 0.0006372
4: 0.0100561, 0.0148711, 0.0101274, 0.0147403, -0.0029950, 0.0030917
5: -0.0027397, -0.0020209, -0.0027201, -0.0020315, -0.0004615, 0.0004471
6: 0.9939595, 0.9952778, 0.9939790, 0.9952420, -0.0008200, 0.0008465
7: 0.0048204, 0.0135364, 0.0049495, 0.0132996, -0.0054214, 0.0055965
8: 0.0024985, 0.0052292, 0.0025390, 0.0051550, -0.0016985, 0.0017533
9: -0.0177659, -0.0123159, -0.0176178, -0.0123966, -0.0034994, 0.0033899

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004824, upper bound: 0.0005563
time: 1.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004816, upper bound: 0.0005563
time: 2.08 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 5.56 seconds
NS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.56
Output dim: 6, lower bound: -0.0004818, upper bound: 0.0005551
NS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.56
Output dim: 6, lower bound: -0.0004958, upper bound: 0.0005551
NS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.56
Output dim: 6, lower bound: -0.0004824, upper bound: 0.0005563
NS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.56
Output dim: 6, lower bound: -0.0004816, upper bound: 0.0005563

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.65 + 134.12 = 137.77 seconds
