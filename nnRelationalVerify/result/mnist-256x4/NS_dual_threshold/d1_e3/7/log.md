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
execution time: IAR + RelationalAnalysis = 1.73 + 2.44 = 4.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0006282, upper bound: 0.0006282

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006091, upper bound: 0.0005775
time: 1.53 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0006091, upper bound: 0.0006091
time: 1.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.22
Output dim: 6, lower bound: -0.0006091, upper bound: 0.0005775
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.22
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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005713, upper bound: 0.0005612
time: 1.55 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005938, upper bound: 0.0005632
time: 1.59 seconds

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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005712, upper bound: 0.0005918
time: 1.54 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005938, upper bound: 0.0005938
time: 1.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.63 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.63
Output dim: 6, lower bound: -0.0005713, upper bound: 0.0005612
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.63
Output dim: 6, lower bound: -0.0005938, upper bound: 0.0005632
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.63
Output dim: 6, lower bound: -0.0005712, upper bound: 0.0005918
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.63
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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005562, upper bound: 0.0005296
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005572, upper bound: 0.0005474
time: 1.49 seconds

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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005799, upper bound: 0.0005301
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005817, upper bound: 0.0005497
time: 1.63 seconds

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005562, upper bound: 0.0005513
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005573, upper bound: 0.0005796
time: 1.67 seconds

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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005799, upper bound: 0.0005522
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005818, upper bound: 0.0005818
time: 1.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.86 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.86
Output dim: 6, lower bound: -0.0005562, upper bound: 0.0005296
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.86
Output dim: 6, lower bound: -0.0005572, upper bound: 0.0005474
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 6, lower bound: -0.0005799, upper bound: 0.0005301
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 6, lower bound: -0.0005817, upper bound: 0.0005497
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.86
Output dim: 6, lower bound: -0.0005562, upper bound: 0.0005513
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 6, lower bound: -0.0005573, upper bound: 0.0005796
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 6, lower bound: -0.0005799, upper bound: 0.0005522
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.86
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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005601, upper bound: 0.0005301
time: 1.59 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005601, upper bound: 0.0005301
time: 2.29 seconds

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

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005623, upper bound: 0.0005497
time: 1.63 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005623, upper bound: 0.0005496
time: 2.31 seconds

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

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
time: 2.40 seconds

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

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005474, upper bound: 0.0005523
time: 1.75 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005474, upper bound: 0.0005522
time: 1.58 seconds

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005817
time: 1.74 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005818
time: 1.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.21 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005601, upper bound: 0.0005301
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005601, upper bound: 0.0005301
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005623, upper bound: 0.0005497
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005623, upper bound: 0.0005496
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005232, upper bound: 0.0005796
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005474, upper bound: 0.0005523
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005474, upper bound: 0.0005522
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.21
Output dim: 6, lower bound: -0.0005497, upper bound: 0.0005817
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.21
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

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005687
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005647
time: 1.58 seconds

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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005108, upper bound: 0.0005636
time: 2.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005647
time: 1.66 seconds

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005352, upper bound: 0.0005709
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005353, upper bound: 0.0005671
time: 1.58 seconds

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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005389, upper bound: 0.0005660
time: 2.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005353, upper bound: 0.0005671
time: 2.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.19 seconds
NS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005687
NS_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 6.19
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005647
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.19
Output dim: 6, lower bound: -0.0005108, upper bound: 0.0005636
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.19
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005647
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 6, lower bound: -0.0005352, upper bound: 0.0005709
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 6, lower bound: -0.0005353, upper bound: 0.0005671
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 6, lower bound: -0.0005389, upper bound: 0.0005660
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.19
Output dim: 6, lower bound: -0.0005353, upper bound: 0.0005671

## BFS NS instance: NS_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0073579, 0.0080853, 0.0073928, 0.0080449, -0.0004619, 0.0004556
1: 0.0023336, 0.0037426, 0.0024012, 0.0036643, -0.0008946, 0.0008825
2: -0.0132598, -0.0018946, -0.0126280, -0.0024400, -0.0071184, 0.0072158
3: -0.0023434, -0.0013283, -0.0022946, -0.0013847, -0.0006445, 0.0006358
4: 0.0099666, 0.0148916, 0.0102030, 0.0146178, -0.0031269, 0.0030847
5: -0.0027427, -0.0020075, -0.0027019, -0.0020428, -0.0004605, 0.0004668
6: 0.9939350, 0.9952835, 0.9939997, 0.9952085, -0.0008561, 0.0008445
7: 0.0046585, 0.0135735, 0.0050863, 0.0130779, -0.0056602, 0.0055838
8: 0.0024478, 0.0052408, 0.0025818, 0.0050856, -0.0017733, 0.0017494
9: -0.0177891, -0.0122146, -0.0174793, -0.0124822, -0.0034915, 0.0035393

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B1_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005677
time: 1.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005687
time: 1.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0073571, 0.0080948, 0.0073888, 0.0080697, -0.0004723, 0.0004624
1: 0.0023322, 0.0037611, 0.0023935, 0.0037124, -0.0009147, 0.0008956
2: -0.0134085, -0.0018830, -0.0130156, -0.0023778, -0.0072240, 0.0073781
3: -0.0023444, -0.0013150, -0.0023002, -0.0013501, -0.0006590, 0.0006452
4: 0.0099616, 0.0149561, 0.0101760, 0.0147858, -0.0031972, 0.0031305
5: -0.0027524, -0.0020068, -0.0027269, -0.0020388, -0.0004673, 0.0004773
6: 0.9939337, 0.9953011, 0.9939924, 0.9952545, -0.0008754, 0.0008571
7: 0.0046494, 0.0136902, 0.0050375, 0.0133820, -0.0057875, 0.0056667
8: 0.0024450, 0.0052774, 0.0025666, 0.0051808, -0.0018132, 0.0017753
9: -0.0178621, -0.0122090, -0.0176694, -0.0124517, -0.0035433, 0.0036189

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005454
time: 2.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005708
time: 2.26 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0073586, 0.0080949, 0.0073909, 0.0080677, -0.0004745, 0.0004625
1: 0.0023350, 0.0037612, 0.0023975, 0.0037084, -0.0009190, 0.0008958
2: -0.0134099, -0.0019060, -0.0129840, -0.0024102, -0.0072254, 0.0074128
3: -0.0023423, -0.0013149, -0.0022973, -0.0013529, -0.0006621, 0.0006453
4: 0.0099716, 0.0149567, 0.0101900, 0.0147721, -0.0032123, 0.0031310
5: -0.0027525, -0.0020083, -0.0027249, -0.0020409, -0.0004674, 0.0004795
6: 0.9939364, 0.9953012, 0.9939963, 0.9952508, -0.0008795, 0.0008572
7: 0.0046674, 0.0136913, 0.0050629, 0.0133573, -0.0058147, 0.0056677
8: 0.0024506, 0.0052777, 0.0025745, 0.0051731, -0.0018217, 0.0017757
9: -0.0178628, -0.0122202, -0.0176539, -0.0124675, -0.0035440, 0.0036359

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005439
time: 2.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005671
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0073672, 0.0080945, 0.0073569, 0.0080923, -0.0004503, 0.0004670
1: 0.0023518, 0.0037605, 0.0023317, 0.0037561, -0.0008722, 0.0009046
2: -0.0134037, -0.0020413, -0.0133683, -0.0018794, -0.0072966, 0.0070349
3: -0.0023303, -0.0013154, -0.0023447, -0.0013186, -0.0006283, 0.0006517
4: 0.0100302, 0.0149540, 0.0099601, 0.0149386, -0.0030485, 0.0031619
5: -0.0027521, -0.0020170, -0.0027498, -0.0020066, -0.0004720, 0.0004551
6: 0.9939525, 0.9953005, 0.9939333, 0.9952963, -0.0008347, 0.0008657
7: 0.0047735, 0.0136864, 0.0046466, 0.0136587, -0.0055183, 0.0057236
8: 0.0024838, 0.0052762, 0.0024441, 0.0052675, -0.0017289, 0.0017932
9: -0.0178597, -0.0122866, -0.0178424, -0.0122072, -0.0035789, 0.0034506

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005108, upper bound: 0.0005439
time: 2.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005108, upper bound: 0.0005660
time: 2.39 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0073696, 0.0080918, 0.0073584, 0.0080924, -0.0004501, 0.0004711
1: 0.0023564, 0.0037551, 0.0023346, 0.0037563, -0.0008718, 0.0009125
2: -0.0133607, -0.0020782, -0.0133697, -0.0019025, -0.0073601, 0.0070315
3: -0.0023270, -0.0013193, -0.0023426, -0.0013185, -0.0006280, 0.0006574
4: 0.0100462, 0.0149354, 0.0099701, 0.0149392, -0.0030470, 0.0031894
5: -0.0027493, -0.0020194, -0.0027499, -0.0020081, -0.0004761, 0.0004549
6: 0.9939568, 0.9952954, 0.9939359, 0.9952965, -0.0008342, 0.0008732
7: 0.0048025, 0.0136527, 0.0046647, 0.0136597, -0.0055156, 0.0057734
8: 0.0024929, 0.0052656, 0.0024498, 0.0052678, -0.0017280, 0.0018088
9: -0.0178387, -0.0123047, -0.0178430, -0.0122185, -0.0036101, 0.0034489

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005439
time: 2.30 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005671
time: 2.45 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.52 seconds
NS_A2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005677
NS_A2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005687
NS_A2_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005454
NS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005708
NS_A2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005439
NS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005671
NS_A2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005108, upper bound: 0.0005439
NS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005108, upper bound: 0.0005660
NS_A2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005439
NS_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.52
Output dim: 6, lower bound: -0.0005106, upper bound: 0.0005671

## BFS NS instance: NS_A2_B1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0073579, 0.0080853, 0.0073946, 0.0080122, -0.0004325, 0.0004678
1: 0.0023336, 0.0037426, 0.0024048, 0.0036010, -0.0008377, 0.0009060
2: -0.0132598, -0.0018946, -0.0121176, -0.0024692, -0.0073077, 0.0067567
3: -0.0023434, -0.0013283, -0.0022920, -0.0014303, -0.0006035, 0.0006527
4: 0.0099666, 0.0148916, 0.0102156, 0.0143966, -0.0029279, 0.0031667
5: -0.0027427, -0.0020075, -0.0026689, -0.0020447, -0.0004727, 0.0004371
6: 0.9939350, 0.9952835, 0.9940032, 0.9951480, -0.0008016, 0.0008670
7: 0.0046585, 0.0135735, 0.0051092, 0.0126776, -0.0053001, 0.0057323
8: 0.0024478, 0.0052408, 0.0025890, 0.0049601, -0.0016605, 0.0017959
9: -0.0177891, -0.0122146, -0.0172289, -0.0124965, -0.0035844, 0.0033141

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005515
time: 1.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005677
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0073579, 0.0080853, 0.0073944, 0.0080376, -0.0004400, 0.0004526
1: 0.0023336, 0.0037426, 0.0024044, 0.0036501, -0.0008523, 0.0008767
2: -0.0132598, -0.0018946, -0.0125136, -0.0024661, -0.0070716, 0.0068748
3: -0.0023434, -0.0013283, -0.0022923, -0.0013949, -0.0006140, 0.0006316
4: 0.0099666, 0.0148916, 0.0102143, 0.0145683, -0.0029791, 0.0030644
5: -0.0027427, -0.0020075, -0.0026945, -0.0020445, -0.0004575, 0.0004447
6: 0.9939350, 0.9952835, 0.9940029, 0.9951950, -0.0008156, 0.0008390
7: 0.0046585, 0.0135735, 0.0051068, 0.0129882, -0.0053927, 0.0055471
8: 0.0024478, 0.0052408, 0.0025883, 0.0050575, -0.0016895, 0.0017379
9: -0.0177891, -0.0122146, -0.0174232, -0.0124950, -0.0034686, 0.0033720

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005524
time: 1.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005687
time: 1.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0073585, 0.0080856, 0.0073888, 0.0080697, -0.0004710, 0.0004434
1: 0.0023349, 0.0037431, 0.0023935, 0.0037124, -0.0009123, 0.0008589
2: -0.0132637, -0.0019053, -0.0130156, -0.0023778, -0.0069279, 0.0073582
3: -0.0023424, -0.0013279, -0.0023002, -0.0013501, -0.0006572, 0.0006188
4: 0.0099713, 0.0148933, 0.0101760, 0.0147858, -0.0031886, 0.0030021
5: -0.0027430, -0.0020082, -0.0027269, -0.0020388, -0.0004482, 0.0004760
6: 0.9939363, 0.9952840, 0.9939924, 0.9952545, -0.0008730, 0.0008220
7: 0.0046669, 0.0135766, 0.0050375, 0.0133820, -0.0057719, 0.0054344
8: 0.0024504, 0.0052418, 0.0025666, 0.0051808, -0.0018083, 0.0017026
9: -0.0177911, -0.0122199, -0.0176694, -0.0124517, -0.0033981, 0.0036091

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005690
time: 2.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005708
time: 2.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0073600, 0.0080857, 0.0073909, 0.0080677, -0.0004732, 0.0004439
1: 0.0023378, 0.0037433, 0.0023975, 0.0037084, -0.0009165, 0.0008597
2: -0.0132651, -0.0019284, -0.0129840, -0.0024102, -0.0069343, 0.0073928
3: -0.0023403, -0.0013278, -0.0022973, -0.0013529, -0.0006603, 0.0006193
4: 0.0099813, 0.0148939, 0.0101900, 0.0147721, -0.0032036, 0.0030049
5: -0.0027431, -0.0020097, -0.0027249, -0.0020409, -0.0004486, 0.0004782
6: 0.9939390, 0.9952840, 0.9939963, 0.9952508, -0.0008771, 0.0008227
7: 0.0046850, 0.0135777, 0.0050629, 0.0133573, -0.0057990, 0.0054394
8: 0.0024561, 0.0052421, 0.0025745, 0.0051731, -0.0018168, 0.0017041
9: -0.0177917, -0.0122312, -0.0176539, -0.0124675, -0.0034012, 0.0036261

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005649
time: 2.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005671
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.0073687, 0.0080853, 0.0073569, 0.0080923, -0.0004491, 0.0004460
1: 0.0023546, 0.0037425, 0.0023317, 0.0037561, -0.0008698, 0.0008639
2: -0.0132588, -0.0020637, -0.0133683, -0.0018794, -0.0069679, 0.0070158
3: -0.0023282, -0.0013284, -0.0023447, -0.0013186, -0.0006266, 0.0006223
4: 0.0100399, 0.0148912, 0.0099601, 0.0149386, -0.0030402, 0.0030194
5: -0.0027427, -0.0020185, -0.0027498, -0.0020066, -0.0004507, 0.0004538
6: 0.9939552, 0.9952834, 0.9939333, 0.9952963, -0.0008324, 0.0008267
7: 0.0047911, 0.0135728, 0.0046466, 0.0136587, -0.0055034, 0.0054657
8: 0.0024894, 0.0052406, 0.0024441, 0.0052675, -0.0017242, 0.0017124
9: -0.0177887, -0.0122976, -0.0178424, -0.0122072, -0.0034177, 0.0034412

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005639
time: 2.19 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005660
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.0073711, 0.0080823, 0.0073584, 0.0080924, -0.0004488, 0.0004528
1: 0.0023592, 0.0037368, 0.0023346, 0.0037563, -0.0008694, 0.0008770
2: -0.0132125, -0.0021010, -0.0133697, -0.0019025, -0.0070738, 0.0070122
3: -0.0023249, -0.0013325, -0.0023426, -0.0013185, -0.0006263, 0.0006318
4: 0.0100561, 0.0148711, 0.0099701, 0.0149392, -0.0030387, 0.0030654
5: -0.0027397, -0.0020209, -0.0027499, -0.0020081, -0.0004576, 0.0004536
6: 0.9939595, 0.9952778, 0.9939359, 0.9952965, -0.0008320, 0.0008393
7: 0.0048204, 0.0135364, 0.0046647, 0.0136597, -0.0055005, 0.0055488
8: 0.0024985, 0.0052292, 0.0024498, 0.0052678, -0.0017233, 0.0017384
9: -0.0177659, -0.0123159, -0.0178430, -0.0122185, -0.0034696, 0.0034394

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of NS_A2_B2_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005649
time: 2.29 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0005002, upper bound: 0.0005671
time: 2.32 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.47 seconds
NS_A2_B1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005515
NS_A2_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005677
NS_A2_B1_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005524
NS_A2_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005687
NS_A2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005690
NS_A2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005708
NS_A2_B2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005649
NS_A2_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005671
NS_A2_B2_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005639
NS_A2_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005007, upper bound: 0.0005660
NS_A2_B2_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005006, upper bound: 0.0005649
NS_A2_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.47
Output dim: 6, lower bound: -0.0005002, upper bound: 0.0005671

## BFS NS instance: NS_A2_B1_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0073585, 0.0080856, 0.0073946, 0.0080122, -0.0004319, 0.0004696
1: 0.0023349, 0.0037431, 0.0024048, 0.0036010, -0.0008365, 0.0009095
2: -0.0132637, -0.0019053, -0.0121176, -0.0024692, -0.0073361, 0.0067473
3: -0.0023424, -0.0013279, -0.0022920, -0.0014303, -0.0006026, 0.0006552
4: 0.0099713, 0.0148933, 0.0102156, 0.0143966, -0.0029239, 0.0031790
5: -0.0027430, -0.0020082, -0.0026689, -0.0020447, -0.0004746, 0.0004365
6: 0.9939363, 0.9952840, 0.9940032, 0.9951480, -0.0008005, 0.0008704
7: 0.0046669, 0.0135766, 0.0051092, 0.0126776, -0.0052927, 0.0057545
8: 0.0024504, 0.0052418, 0.0025890, 0.0049601, -0.0016582, 0.0018029
9: -0.0177911, -0.0122199, -0.0172289, -0.0124965, -0.0035983, 0.0033095

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of NS_A2_B1_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004893, upper bound: 0.0005333
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004893, upper bound: 0.0005566
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0073585, 0.0080856, 0.0073944, 0.0080376, -0.0004395, 0.0004515
1: 0.0023349, 0.0037431, 0.0024044, 0.0036501, -0.0008512, 0.0008744
2: -0.0132637, -0.0019053, -0.0125136, -0.0024661, -0.0070530, 0.0068658
3: -0.0023424, -0.0013279, -0.0022923, -0.0013949, -0.0006132, 0.0006299
4: 0.0099713, 0.0148933, 0.0102143, 0.0145683, -0.0029752, 0.0030563
5: -0.0027430, -0.0020082, -0.0026945, -0.0020445, -0.0004562, 0.0004441
6: 0.9939363, 0.9952840, 0.9940029, 0.9951950, -0.0008146, 0.0008368
7: 0.0046669, 0.0135766, 0.0051068, 0.0129882, -0.0053857, 0.0055325
8: 0.0024504, 0.0052418, 0.0025883, 0.0050575, -0.0016873, 0.0017333
9: -0.0177911, -0.0122199, -0.0174232, -0.0124950, -0.0034594, 0.0033676

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of NS_A2_B1_A2_B1_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004779, upper bound: 0.0005575
time: 1.75 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004912, upper bound: 0.0005575
time: 1.91 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0073585, 0.0080856, 0.0073921, 0.0080358, -0.0004405, 0.0004537
1: 0.0023349, 0.0037431, 0.0024000, 0.0036467, -0.0008533, 0.0008787
2: -0.0132637, -0.0019053, -0.0124862, -0.0024303, -0.0070877, 0.0068827
3: -0.0023424, -0.0013279, -0.0022955, -0.0013974, -0.0006147, 0.0006330
4: 0.0099713, 0.0148933, 0.0101988, 0.0145564, -0.0029825, 0.0030714
5: -0.0027430, -0.0020082, -0.0026927, -0.0020422, -0.0004585, 0.0004452
6: 0.9939363, 0.9952840, 0.9939985, 0.9951917, -0.0008166, 0.0008409
7: 0.0046669, 0.0135766, 0.0050787, 0.0129667, -0.0053989, 0.0055597
8: 0.0024504, 0.0052418, 0.0025795, 0.0050507, -0.0016914, 0.0017418
9: -0.0177911, -0.0122199, -0.0174097, -0.0124774, -0.0034765, 0.0033759

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of NS_A2_B2_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004893, upper bound: 0.0005349
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005582
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073585, 0.0080856, 0.0073905, 0.0080626, -0.0004534, 0.0004404
1: 0.0023349, 0.0037431, 0.0023969, 0.0036986, -0.0008783, 0.0008530
2: -0.0132637, -0.0019053, -0.0129046, -0.0024049, -0.0068798, 0.0070839
3: -0.0023424, -0.0013279, -0.0022978, -0.0013600, -0.0006327, 0.0006145
4: 0.0099713, 0.0148933, 0.0101878, 0.0147377, -0.0030697, 0.0029813
5: -0.0027430, -0.0020082, -0.0027198, -0.0020406, -0.0004450, 0.0004582
6: 0.9939363, 0.9952840, 0.9939956, 0.9952413, -0.0008405, 0.0008162
7: 0.0046669, 0.0135766, 0.0050588, 0.0132949, -0.0055568, 0.0053966
8: 0.0024504, 0.0052418, 0.0025732, 0.0051536, -0.0017409, 0.0016907
9: -0.0177911, -0.0122199, -0.0176149, -0.0124650, -0.0033745, 0.0034746

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of NS_A2_B2_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005367
time: 1.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005599
time: 1.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073600, 0.0080857, 0.0073926, 0.0080604, -0.0004600, 0.0004408
1: 0.0023378, 0.0037433, 0.0024009, 0.0036943, -0.0008910, 0.0008537
2: -0.0132651, -0.0019284, -0.0128697, -0.0024377, -0.0068861, 0.0071871
3: -0.0023403, -0.0013278, -0.0022948, -0.0013631, -0.0006419, 0.0006150
4: 0.0099813, 0.0148939, 0.0102020, 0.0147226, -0.0031144, 0.0029840
5: -0.0027431, -0.0020097, -0.0027175, -0.0020427, -0.0004455, 0.0004649
6: 0.9939390, 0.9952840, 0.9939995, 0.9952372, -0.0008527, 0.0008170
7: 0.0046850, 0.0135777, 0.0050845, 0.0132675, -0.0056377, 0.0054016
8: 0.0024561, 0.0052421, 0.0025813, 0.0051450, -0.0017662, 0.0016923
9: -0.0177917, -0.0122312, -0.0175978, -0.0124810, -0.0033776, 0.0035252

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004783, upper bound: 0.0005563
time: 2.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005563
time: 1.89 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073687, 0.0080853, 0.0073585, 0.0080856, -0.0004316, 0.0004430
1: 0.0023546, 0.0037425, 0.0023349, 0.0037431, -0.0008360, 0.0008581
2: -0.0132588, -0.0020637, -0.0132637, -0.0019053, -0.0069211, 0.0067429
3: -0.0023282, -0.0013284, -0.0023424, -0.0013279, -0.0006022, 0.0006182
4: 0.0100399, 0.0148912, 0.0099713, 0.0148933, -0.0029220, 0.0029992
5: -0.0027427, -0.0020185, -0.0027430, -0.0020082, -0.0004477, 0.0004362
6: 0.9939552, 0.9952834, 0.9939363, 0.9952840, -0.0008000, 0.0008211
7: 0.0047911, 0.0135728, 0.0046669, 0.0135766, -0.0052892, 0.0054290
8: 0.0024894, 0.0052406, 0.0024504, 0.0052418, -0.0016571, 0.0017009
9: -0.0177887, -0.0122976, -0.0177911, -0.0122199, -0.0033947, 0.0033073

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005342
time: 2.50 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005551
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0073711, 0.0080823, 0.0073600, 0.0080857, -0.0004322, 0.0004498
1: 0.0023592, 0.0037368, 0.0023378, 0.0037433, -0.0008371, 0.0008713
2: -0.0132125, -0.0021010, -0.0132651, -0.0019284, -0.0070279, 0.0067518
3: -0.0023249, -0.0013325, -0.0023403, -0.0013278, -0.0006030, 0.0006277
4: 0.0100561, 0.0148711, 0.0099813, 0.0148939, -0.0029258, 0.0030454
5: -0.0027397, -0.0020209, -0.0027431, -0.0020097, -0.0004546, 0.0004368
6: 0.9939595, 0.9952778, 0.9939390, 0.9952840, -0.0008011, 0.0008338
7: 0.0048204, 0.0135364, 0.0046850, 0.0135777, -0.0052963, 0.0055128
8: 0.0024985, 0.0052292, 0.0024561, 0.0052421, -0.0016593, 0.0017271
9: -0.0177659, -0.0123159, -0.0177917, -0.0122312, -0.0034471, 0.0033117

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of NS_A2_B2_A2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005345
time: 2.44 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0004913, upper bound: 0.0005563
time: 2.23 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 6.38 seconds
NS_A2_B1_A2_B1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004893, upper bound: 0.0005333
NS_A2_B1_A2_B1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004893, upper bound: 0.0005566
NS_A2_B1_A2_B1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004779, upper bound: 0.0005575
NS_A2_B1_A2_B1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004912, upper bound: 0.0005575
NS_A2_B2_A2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004893, upper bound: 0.0005349
NS_A2_B2_A2_B1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005582
NS_A2_B2_A2_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005367
NS_A2_B2_A2_B1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005599
NS_A2_B2_A2_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004783, upper bound: 0.0005563
NS_A2_B2_A2_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005563
NS_A2_B2_A2_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005342
NS_A2_B2_A2_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005551
NS_A2_B2_A2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004918, upper bound: 0.0005345
NS_A2_B2_A2_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.38
Output dim: 6, lower bound: -0.0004913, upper bound: 0.0005563

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.17 + 187.23 = 191.39 seconds
