## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000280174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0009523, 0.0011007, 0.0009523, 0.0011007, -0.0001450, 0.0001450)
1: (0.9936138, 0.9940606, 0.9936138, 0.9940606, -0.0004041, 0.0004041)
2: (-0.0066219, -0.0050497, -0.0066219, -0.0050497, -0.0013876, 0.0013876)
3: (0.0038073, 0.0040181, 0.0038073, 0.0040181, -0.0001820, 0.0001820)
4: (0.0024080, 0.0036506, 0.0024080, 0.0036506, -0.0011624, 0.0011624)
5: (0.0060248, 0.0065048, 0.0060248, 0.0065048, -0.0004801, 0.0004801)
6: (-0.0014196, -0.0008739, -0.0014196, -0.0008739, -0.0004778, 0.0004778)
7: (-0.0083167, -0.0079503, -0.0083167, -0.0079503, -0.0003664, 0.0003664)
8: (0.0048863, 0.0069520, 0.0048863, 0.0069520, -0.0016405, 0.0016405)
9: (-0.0036874, -0.0032290, -0.0036874, -0.0032290, -0.0004584, 0.0004584)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.19 + 1.51 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0003743, upper bound: 0.0003743

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003718, upper bound: 0.0003716
time: 0.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003724, upper bound: 0.0003724
time: 0.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 1, lower bound: -0.0003718, upper bound: 0.0003716
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 1, lower bound: -0.0003724, upper bound: 0.0003724

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0009518, 0.0011005, 0.0009523, 0.0011007, -0.0001454, 0.0001447
1: 0.9936161, 0.9940617, 0.9936141, 0.9940604, -0.0004019, 0.0004047
2: -0.0065980, -0.0050475, -0.0066194, -0.0050500, -0.0013628, 0.0013834
3: 0.0038068, 0.0040165, 0.0038074, 0.0040179, -0.0001820, 0.0001804
4: 0.0024063, 0.0036317, 0.0024083, 0.0036486, -0.0011591, 0.0011435
5: 0.0060236, 0.0065053, 0.0060250, 0.0065048, -0.0004812, 0.0004804
6: -0.0014113, -0.0008731, -0.0014187, -0.0008740, -0.0004692, 0.0004763
7: -0.0083171, -0.0079494, -0.0083166, -0.0079504, -0.0003667, 0.0003672
8: 0.0048835, 0.0069206, 0.0048867, 0.0069487, -0.0016349, 0.0016105
9: -0.0036874, -0.0032274, -0.0036874, -0.0032292, -0.0004582, 0.0004600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003336, upper bound: 0.0003418
time: 0.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003686, upper bound: 0.0003684
time: 0.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0009529, 0.0011006, 0.0009523, 0.0011007, -0.0001444, 0.0001448
1: 0.9936160, 0.9940590, 0.9936139, 0.9940606, -0.0004020, 0.0004026
2: -0.0065994, -0.0050522, -0.0066212, -0.0050498, -0.0013638, 0.0013847
3: 0.0038079, 0.0040166, 0.0038073, 0.0040180, -0.0001813, 0.0001804
4: 0.0024100, 0.0036328, 0.0024081, 0.0036501, -0.0011601, 0.0011439
5: 0.0060262, 0.0065043, 0.0060248, 0.0065048, -0.0004786, 0.0004795
6: -0.0014118, -0.0008748, -0.0014194, -0.0008739, -0.0004695, 0.0004768
7: -0.0083162, -0.0079513, -0.0083167, -0.0079503, -0.0003659, 0.0003654
8: 0.0048896, 0.0069224, 0.0048864, 0.0069511, -0.0016366, 0.0016090
9: -0.0036873, -0.0032308, -0.0036874, -0.0032291, -0.0004583, 0.0004565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003350, upper bound: 0.0003428
time: 0.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003692, upper bound: 0.0003692
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.25 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 1, lower bound: -0.0003336, upper bound: 0.0003418
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 1, lower bound: -0.0003686, upper bound: 0.0003684
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 1, lower bound: -0.0003350, upper bound: 0.0003428
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 1, lower bound: -0.0003692, upper bound: 0.0003692

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009535, 0.0011005, 0.0009653, 0.0011011, -0.0001440, 0.0001318
1: 0.9936161, 0.9940574, 0.9936093, 0.9940277, -0.0003689, 0.0004040
2: -0.0065980, -0.0050553, -0.0066693, -0.0051068, -0.0012822, 0.0013828
3: 0.0038086, 0.0040165, 0.0038209, 0.0040212, -0.0001827, 0.0001663
4: 0.0024124, 0.0036317, 0.0024531, 0.0036880, -0.0011491, 0.0010785
5: 0.0060279, 0.0065036, 0.0060567, 0.0064925, -0.0004646, 0.0004470
6: -0.0014113, -0.0008758, -0.0014361, -0.0008937, -0.0004413, 0.0004766
7: -0.0083156, -0.0079525, -0.0083059, -0.0079732, -0.0003424, 0.0003534
8: 0.0048936, 0.0069206, 0.0049613, 0.0070142, -0.0016668, 0.0015070
9: -0.0036873, -0.0032331, -0.0036868, -0.0032706, -0.0004167, 0.0004537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002907, upper bound: 0.0002741
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003309, upper bound: 0.0003391
time: 0.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009518, 0.0011005, 0.0009552, 0.0011007, -0.0001454, 0.0001422
1: 0.9936161, 0.9940617, 0.9936141, 0.9940534, -0.0003980, 0.0004041
2: -0.0065980, -0.0050475, -0.0066194, -0.0050624, -0.0013645, 0.0013723
3: 0.0038068, 0.0040165, 0.0038103, 0.0040179, -0.0001816, 0.0001794
4: 0.0024063, 0.0036317, 0.0024180, 0.0036486, -0.0011483, 0.0011406
5: 0.0060236, 0.0065053, 0.0060318, 0.0065021, -0.0004785, 0.0004735
6: -0.0014113, -0.0008731, -0.0014187, -0.0008783, -0.0004700, 0.0004726
7: -0.0083171, -0.0079494, -0.0083143, -0.0079554, -0.0003617, 0.0003649
8: 0.0048835, 0.0069206, 0.0049030, 0.0069487, -0.0016261, 0.0016208
9: -0.0036874, -0.0032274, -0.0036872, -0.0032382, -0.0004492, 0.0004598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003372, upper bound: 0.0003165
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003658, upper bound: 0.0003657
time: 0.54 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009546, 0.0011006, 0.0009652, 0.0011011, -0.0001429, 0.0001319
1: 0.9936160, 0.9940544, 0.9936091, 0.9940278, -0.0003691, 0.0004019
2: -0.0065994, -0.0050601, -0.0066711, -0.0051065, -0.0012866, 0.0013836
3: 0.0038098, 0.0040166, 0.0038208, 0.0040213, -0.0001820, 0.0001663
4: 0.0024163, 0.0036328, 0.0024529, 0.0036895, -0.0011497, 0.0010817
5: 0.0060306, 0.0065026, 0.0060565, 0.0064926, -0.0004620, 0.0004461
6: -0.0014118, -0.0008775, -0.0014367, -0.0008936, -0.0004428, 0.0004769
7: -0.0083147, -0.0079545, -0.0083059, -0.0079731, -0.0003416, 0.0003515
8: 0.0049000, 0.0069224, 0.0049610, 0.0070166, -0.0016669, 0.0015103
9: -0.0036873, -0.0032366, -0.0036868, -0.0032704, -0.0004168, 0.0004502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002948, upper bound: 0.0002784
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003323, upper bound: 0.0003400
time: 0.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009529, 0.0011006, 0.0009551, 0.0011007, -0.0001443, 0.0001422
1: 0.9936160, 0.9940590, 0.9936139, 0.9940533, -0.0003979, 0.0004020
2: -0.0065994, -0.0050522, -0.0066212, -0.0050621, -0.0013669, 0.0013736
3: 0.0038079, 0.0040166, 0.0038103, 0.0040180, -0.0001809, 0.0001793
4: 0.0024100, 0.0036328, 0.0024178, 0.0036501, -0.0011493, 0.0011430
5: 0.0060262, 0.0065043, 0.0060317, 0.0065022, -0.0004760, 0.0004726
6: -0.0014118, -0.0008748, -0.0014194, -0.0008782, -0.0004708, 0.0004730
7: -0.0083162, -0.0079513, -0.0083143, -0.0079553, -0.0003609, 0.0003630
8: 0.0048896, 0.0069224, 0.0049026, 0.0069511, -0.0016277, 0.0016209
9: -0.0036873, -0.0032308, -0.0036872, -0.0032381, -0.0004493, 0.0004564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003390, upper bound: 0.0003181
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003664, upper bound: 0.0003664
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.25 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 1, lower bound: -0.0002907, upper bound: 0.0002741
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 1, lower bound: -0.0003309, upper bound: 0.0003391
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 1, lower bound: -0.0003372, upper bound: 0.0003165
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 1, lower bound: -0.0003658, upper bound: 0.0003657
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 1, lower bound: -0.0002948, upper bound: 0.0002784
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 1, lower bound: -0.0003323, upper bound: 0.0003400
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 1, lower bound: -0.0003390, upper bound: 0.0003181
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 1, lower bound: -0.0003664, upper bound: 0.0003664

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0009690, 0.0011011, 0.0009673, 0.0011011, -0.0001286, 0.0001303
1: 0.9936086, 0.9940183, 0.9936093, 0.9940226, -0.0003697, 0.0003655
2: -0.0066771, -0.0051233, -0.0066693, -0.0051155, -0.0013178, 0.0012951
3: 0.0038248, 0.0040217, 0.0038230, 0.0040212, -0.0001665, 0.0001682
4: 0.0024662, 0.0036942, 0.0024600, 0.0036881, -0.0010787, 0.0010973
5: 0.0060659, 0.0064889, 0.0060615, 0.0064906, -0.0004248, 0.0004274
6: -0.0014388, -0.0008994, -0.0014361, -0.0008967, -0.0004541, 0.0004462
7: -0.0083028, -0.0079799, -0.0083042, -0.0079767, -0.0003260, 0.0003244
8: 0.0049830, 0.0070245, 0.0049727, 0.0070142, -0.0015537, 0.0015725
9: -0.0036866, -0.0032827, -0.0036867, -0.0032770, -0.0004096, 0.0004040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002879, upper bound: 0.0002719
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002879, upper bound: 0.0002741
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0009559, 0.0011005, 0.0009653, 0.0011011, -0.0001416, 0.0001318
1: 0.9936161, 0.9940512, 0.9936093, 0.9940277, -0.0003689, 0.0003976
2: -0.0065980, -0.0050656, -0.0066693, -0.0051068, -0.0012798, 0.0013580
3: 0.0038111, 0.0040165, 0.0038209, 0.0040212, -0.0001799, 0.0001663
4: 0.0024206, 0.0036317, 0.0024531, 0.0036880, -0.0011310, 0.0010749
5: 0.0060336, 0.0065014, 0.0060567, 0.0064925, -0.0004589, 0.0004448
6: -0.0014113, -0.0008794, -0.0014361, -0.0008937, -0.0004405, 0.0004679
7: -0.0083137, -0.0079567, -0.0083059, -0.0079732, -0.0003404, 0.0003492
8: 0.0049072, 0.0069206, 0.0049613, 0.0070142, -0.0016295, 0.0015068
9: -0.0036872, -0.0032406, -0.0036868, -0.0032706, -0.0004166, 0.0004462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003289, upper bound: 0.0003385
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003289, upper bound: 0.0003391
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0009671, 0.0011011, 0.0009571, 0.0011007, -0.0001301, 0.0001407
1: 0.9936086, 0.9940228, 0.9936141, 0.9940482, -0.0003984, 0.0003659
2: -0.0066771, -0.0051149, -0.0066194, -0.0050708, -0.0013931, 0.0012849
3: 0.0038228, 0.0040217, 0.0038123, 0.0040179, -0.0001655, 0.0001811
4: 0.0024596, 0.0036942, 0.0024247, 0.0036486, -0.0010781, 0.0011575
5: 0.0060612, 0.0064907, 0.0060366, 0.0065003, -0.0004391, 0.0004542
6: -0.0014388, -0.0008965, -0.0014187, -0.0008812, -0.0004804, 0.0004423
7: -0.0083043, -0.0079765, -0.0083127, -0.0079588, -0.0003456, 0.0003361
8: 0.0049721, 0.0070245, 0.0049141, 0.0069487, -0.0015134, 0.0016780
9: -0.0036867, -0.0032766, -0.0036872, -0.0032444, -0.0004423, 0.0004106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003368, upper bound: 0.0003164
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003368, upper bound: 0.0003165
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0009541, 0.0011005, 0.0009552, 0.0011007, -0.0001430, 0.0001422
1: 0.9936161, 0.9940560, 0.9936141, 0.9940534, -0.0003980, 0.0003981
2: -0.0065980, -0.0050578, -0.0066194, -0.0050624, -0.0013621, 0.0013501
3: 0.0038092, 0.0040165, 0.0038103, 0.0040179, -0.0001790, 0.0001794
4: 0.0024144, 0.0036317, 0.0024180, 0.0036486, -0.0011306, 0.0011372
5: 0.0060293, 0.0065031, 0.0060318, 0.0065021, -0.0004728, 0.0004713
6: -0.0014113, -0.0008767, -0.0014187, -0.0008783, -0.0004692, 0.0004648
7: -0.0083151, -0.0079535, -0.0083143, -0.0079554, -0.0003598, 0.0003608
8: 0.0048970, 0.0069206, 0.0049030, 0.0069487, -0.0015948, 0.0016206
9: -0.0036873, -0.0032349, -0.0036872, -0.0032382, -0.0004491, 0.0004523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003175, upper bound: 0.0003370
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003175, upper bound: 0.0003657
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0009703, 0.0011012, 0.0009672, 0.0011011, -0.0001273, 0.0001303
1: 0.9936085, 0.9940149, 0.9936091, 0.9940229, -0.0003701, 0.0003624
2: -0.0066783, -0.0051290, -0.0066711, -0.0051152, -0.0013204, 0.0012957
3: 0.0038262, 0.0040218, 0.0038229, 0.0040213, -0.0001653, 0.0001684
4: 0.0024707, 0.0036951, 0.0024598, 0.0036895, -0.0010792, 0.0010988
5: 0.0060691, 0.0064877, 0.0060614, 0.0064907, -0.0004216, 0.0004263
6: -0.0014392, -0.0009014, -0.0014367, -0.0008966, -0.0004550, 0.0004465
7: -0.0083017, -0.0079822, -0.0083043, -0.0079766, -0.0003250, 0.0003221
8: 0.0049905, 0.0070261, 0.0049724, 0.0070166, -0.0015535, 0.0015761
9: -0.0036866, -0.0032869, -0.0036867, -0.0032768, -0.0004098, 0.0003999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002892, upper bound: 0.0002743
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002892, upper bound: 0.0002784
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0009569, 0.0011006, 0.0009652, 0.0011011, -0.0001406, 0.0001319
1: 0.9936160, 0.9940488, 0.9936091, 0.9940278, -0.0003691, 0.0003951
2: -0.0065994, -0.0050702, -0.0066711, -0.0051065, -0.0012841, 0.0013579
3: 0.0038122, 0.0040166, 0.0038208, 0.0040213, -0.0001790, 0.0001663
4: 0.0024242, 0.0036328, 0.0024529, 0.0036895, -0.0011309, 0.0010781
5: 0.0060362, 0.0065004, 0.0060565, 0.0064926, -0.0004564, 0.0004439
6: -0.0014118, -0.0008810, -0.0014367, -0.0008936, -0.0004420, 0.0004679
7: -0.0083128, -0.0079585, -0.0083059, -0.0079731, -0.0003397, 0.0003474
8: 0.0049132, 0.0069224, 0.0049610, 0.0070166, -0.0016288, 0.0015101
9: -0.0036872, -0.0032439, -0.0036868, -0.0032704, -0.0004167, 0.0004429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003291, upper bound: 0.0003387
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003291, upper bound: 0.0003397
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0009685, 0.0011012, 0.0009570, 0.0011007, -0.0001288, 0.0001407
1: 0.9936085, 0.9940197, 0.9936139, 0.9940486, -0.0003985, 0.0003628
2: -0.0066783, -0.0051207, -0.0066212, -0.0050706, -0.0013950, 0.0012853
3: 0.0038242, 0.0040218, 0.0038123, 0.0040180, -0.0001643, 0.0001811
4: 0.0024642, 0.0036952, 0.0024245, 0.0036501, -0.0010784, 0.0011593
5: 0.0060645, 0.0064895, 0.0060364, 0.0065003, -0.0004359, 0.0004530
6: -0.0014392, -0.0008985, -0.0014194, -0.0008811, -0.0004810, 0.0004425
7: -0.0083032, -0.0079788, -0.0083127, -0.0079587, -0.0003446, 0.0003339
8: 0.0049797, 0.0070261, 0.0049138, 0.0069511, -0.0015139, 0.0016804
9: -0.0036866, -0.0032808, -0.0036872, -0.0032442, -0.0004424, 0.0004063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003370, upper bound: 0.0003175
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003370, upper bound: 0.0003181
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0009552, 0.0011006, 0.0009551, 0.0011007, -0.0001420, 0.0001422
1: 0.9936160, 0.9940532, 0.9936139, 0.9940533, -0.0003979, 0.0003957
2: -0.0065994, -0.0050624, -0.0066212, -0.0050621, -0.0013644, 0.0013500
3: 0.0038103, 0.0040166, 0.0038103, 0.0040180, -0.0001781, 0.0001793
4: 0.0024181, 0.0036328, 0.0024178, 0.0036501, -0.0011305, 0.0011394
5: 0.0060319, 0.0065021, 0.0060317, 0.0065022, -0.0004703, 0.0004704
6: -0.0014118, -0.0008783, -0.0014194, -0.0008782, -0.0004700, 0.0004648
7: -0.0083143, -0.0079554, -0.0083143, -0.0079553, -0.0003590, 0.0003589
8: 0.0049030, 0.0069224, 0.0049026, 0.0069511, -0.0015945, 0.0016207
9: -0.0036872, -0.0032383, -0.0036872, -0.0032381, -0.0004492, 0.0004490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003657, upper bound: 0.0003658
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003657, upper bound: 0.0003664
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.35 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0002879, upper bound: 0.0002719
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0002879, upper bound: 0.0002741
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003289, upper bound: 0.0003385
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003289, upper bound: 0.0003391
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003368, upper bound: 0.0003164
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003368, upper bound: 0.0003165
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003175, upper bound: 0.0003370
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003175, upper bound: 0.0003657
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0002892, upper bound: 0.0002743
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0002892, upper bound: 0.0002784
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003291, upper bound: 0.0003387
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003291, upper bound: 0.0003397
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003370, upper bound: 0.0003175
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003370, upper bound: 0.0003181
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003657, upper bound: 0.0003658
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 1, lower bound: -0.0003657, upper bound: 0.0003664

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009690, 0.0011011, 0.0009665, 0.0011009, -0.0001284, 0.0001310
1: 0.9936086, 0.9940183, 0.9936113, 0.9940248, -0.0003714, 0.0003638
2: -0.0066771, -0.0051233, -0.0066482, -0.0051120, -0.0013239, 0.0012739
3: 0.0038248, 0.0040217, 0.0038222, 0.0040198, -0.0001653, 0.0001688
4: 0.0024662, 0.0036942, 0.0024573, 0.0036714, -0.0010619, 0.0011021
5: 0.0060659, 0.0064889, 0.0060596, 0.0064914, -0.0004255, 0.0004294
6: -0.0014388, -0.0008994, -0.0014287, -0.0008955, -0.0004562, 0.0004391
7: -0.0083028, -0.0079799, -0.0083049, -0.0079754, -0.0003274, 0.0003250
8: 0.0049830, 0.0070245, 0.0049682, 0.0069865, -0.0015299, 0.0015805
9: -0.0036866, -0.0032827, -0.0036867, -0.0032745, -0.0004122, 0.0004041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.17 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002574, upper bound: 0.0002232
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002589, upper bound: 0.0002445
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009690, 0.0011011, 0.0009678, 0.0011009, -0.0001284, 0.0001297
1: 0.9936086, 0.9940183, 0.9936112, 0.9940214, -0.0003684, 0.0003634
2: -0.0066771, -0.0051233, -0.0066492, -0.0051178, -0.0013162, 0.0012781
3: 0.0038248, 0.0040217, 0.0038235, 0.0040199, -0.0001651, 0.0001677
4: 0.0024662, 0.0036942, 0.0024619, 0.0036722, -0.0010655, 0.0010960
5: 0.0060659, 0.0064889, 0.0060628, 0.0064901, -0.0004243, 0.0004261
6: -0.0014388, -0.0008994, -0.0014291, -0.0008975, -0.0004535, 0.0004404
7: -0.0083028, -0.0079799, -0.0083038, -0.0079777, -0.0003251, 0.0003239
8: 0.0049830, 0.0070245, 0.0049758, 0.0069878, -0.0015247, 0.0015703
9: -0.0036866, -0.0032827, -0.0036867, -0.0032787, -0.0004079, 0.0004040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.19 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002574, upper bound: 0.0002245
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002589, upper bound: 0.0002465
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009559, 0.0011005, 0.0009645, 0.0011009, -0.0001415, 0.0001325
1: 0.9936161, 0.9940512, 0.9936113, 0.9940295, -0.0003705, 0.0003958
2: -0.0065980, -0.0050656, -0.0066482, -0.0051035, -0.0012860, 0.0013375
3: 0.0038111, 0.0040165, 0.0038201, 0.0040198, -0.0001786, 0.0001668
4: 0.0024206, 0.0036317, 0.0024505, 0.0036714, -0.0011156, 0.0010798
5: 0.0060336, 0.0065014, 0.0060548, 0.0064932, -0.0004596, 0.0004466
6: -0.0014113, -0.0008794, -0.0014287, -0.0008926, -0.0004427, 0.0004607
7: -0.0083137, -0.0079567, -0.0083065, -0.0079719, -0.0003418, 0.0003499
8: 0.0049072, 0.0069206, 0.0049570, 0.0069865, -0.0016051, 0.0015149
9: -0.0036872, -0.0032406, -0.0036868, -0.0032682, -0.0004190, 0.0004463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003207, upper bound: 0.0003207
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003207, upper bound: 0.0003385
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009559, 0.0011005, 0.0009658, 0.0011009, -0.0001414, 0.0001313
1: 0.9936161, 0.9940512, 0.9936112, 0.9940263, -0.0003676, 0.0003957
2: -0.0065980, -0.0050656, -0.0066492, -0.0051090, -0.0012782, 0.0013398
3: 0.0038111, 0.0040165, 0.0038215, 0.0040199, -0.0001785, 0.0001657
4: 0.0024206, 0.0036317, 0.0024549, 0.0036722, -0.0011170, 0.0010737
5: 0.0060336, 0.0065014, 0.0060579, 0.0064920, -0.0004584, 0.0004435
6: -0.0014113, -0.0008794, -0.0014291, -0.0008945, -0.0004400, 0.0004616
7: -0.0083137, -0.0079567, -0.0083055, -0.0079741, -0.0003395, 0.0003488
8: 0.0049072, 0.0069206, 0.0049643, 0.0069878, -0.0016031, 0.0015047
9: -0.0036872, -0.0032406, -0.0036868, -0.0032723, -0.0004149, 0.0004462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003207, upper bound: 0.0003215
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003207, upper bound: 0.0003391
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009671, 0.0011011, 0.0009565, 0.0011005, -0.0001300, 0.0001413
1: 0.9936086, 0.9940228, 0.9936161, 0.9940498, -0.0003997, 0.0003641
2: -0.0066771, -0.0051149, -0.0065980, -0.0050682, -0.0014003, 0.0012630
3: 0.0038228, 0.0040217, 0.0038117, 0.0040165, -0.0001642, 0.0001815
4: 0.0024596, 0.0036942, 0.0024226, 0.0036317, -0.0010618, 0.0011633
5: 0.0060612, 0.0064907, 0.0060351, 0.0065008, -0.0004396, 0.0004556
6: -0.0014388, -0.0008965, -0.0014113, -0.0008803, -0.0004829, 0.0004347
7: -0.0083043, -0.0079765, -0.0083132, -0.0079577, -0.0003466, 0.0003366
8: 0.0049721, 0.0070245, 0.0049106, 0.0069206, -0.0014882, 0.0016875
9: -0.0036867, -0.0032766, -0.0036872, -0.0032425, -0.0004442, 0.0004106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002658, upper bound: 0.0002691
time: 0.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.67 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003067, upper bound: 0.0002634
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003144, upper bound: 0.0002927
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009671, 0.0011011, 0.0009576, 0.0011006, -0.0001300, 0.0001402
1: 0.9936086, 0.9940228, 0.9936160, 0.9940471, -0.0003971, 0.0003641
2: -0.0066771, -0.0051149, -0.0065994, -0.0050730, -0.0013917, 0.0012680
3: 0.0038228, 0.0040217, 0.0038128, 0.0040166, -0.0001643, 0.0001806
4: 0.0024596, 0.0036942, 0.0024265, 0.0036328, -0.0010652, 0.0011564
5: 0.0060612, 0.0064907, 0.0060378, 0.0064998, -0.0004386, 0.0004529
6: -0.0014388, -0.0008965, -0.0014118, -0.0008820, -0.0004799, 0.0004364
7: -0.0083043, -0.0079765, -0.0083123, -0.0079597, -0.0003447, 0.0003357
8: 0.0049721, 0.0070245, 0.0049170, 0.0069224, -0.0014899, 0.0016761
9: -0.0036867, -0.0032766, -0.0036871, -0.0032460, -0.0004407, 0.0004105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002658, upper bound: 0.0002691
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.66 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003067, upper bound: 0.0002634
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003144, upper bound: 0.0002928
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009541, 0.0011005, 0.0009708, 0.0011013, -0.0001436, 0.0001266
1: 0.9936161, 0.9940560, 0.9936066, 0.9940138, -0.0003583, 0.0004043
2: -0.0065980, -0.0050578, -0.0066977, -0.0051309, -0.0012749, 0.0014129
3: 0.0038092, 0.0040165, 0.0038266, 0.0040231, -0.0001834, 0.0001626
4: 0.0024144, 0.0036317, 0.0024722, 0.0037105, -0.0011713, 0.0010682
5: 0.0060293, 0.0065031, 0.0060701, 0.0064873, -0.0004580, 0.0004330
6: -0.0014113, -0.0008767, -0.0014459, -0.0009021, -0.0004389, 0.0004871
7: -0.0083151, -0.0079535, -0.0083013, -0.0079829, -0.0003322, 0.0003478
8: 0.0048970, 0.0069206, 0.0049930, 0.0070516, -0.0017003, 0.0015060
9: -0.0036873, -0.0032349, -0.0036865, -0.0032882, -0.0003991, 0.0004516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002192, upper bound: 0.0002411
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003173, upper bound: 0.0003368
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009541, 0.0011005, 0.0009575, 0.0011007, -0.0001430, 0.0001397
1: 0.9936161, 0.9940560, 0.9936141, 0.9940473, -0.0003906, 0.0003981
2: -0.0065980, -0.0050578, -0.0066194, -0.0050725, -0.0013290, 0.0013479
3: 0.0038092, 0.0040165, 0.0038127, 0.0040179, -0.0001790, 0.0001759
4: 0.0024144, 0.0036317, 0.0024261, 0.0036486, -0.0011275, 0.0011153
5: 0.0060293, 0.0065031, 0.0060375, 0.0064999, -0.0004706, 0.0004656
6: -0.0014113, -0.0008767, -0.0014187, -0.0008818, -0.0004576, 0.0004641
7: -0.0083151, -0.0079535, -0.0083124, -0.0079595, -0.0003557, 0.0003588
8: 0.0048970, 0.0069206, 0.0049163, 0.0069487, -0.0015945, 0.0015755
9: -0.0036873, -0.0032349, -0.0036871, -0.0032457, -0.0004416, 0.0004522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002192, upper bound: 0.0002957
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003173, upper bound: 0.0003655
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009703, 0.0011012, 0.0009665, 0.0011009, -0.0001271, 0.0001311
1: 0.9936085, 0.9940149, 0.9936113, 0.9940248, -0.0003718, 0.0003605
2: -0.0066783, -0.0051290, -0.0066482, -0.0051120, -0.0013267, 0.0012723
3: 0.0038262, 0.0040218, 0.0038222, 0.0040198, -0.0001640, 0.0001691
4: 0.0024707, 0.0036951, 0.0024573, 0.0036714, -0.0010607, 0.0011046
5: 0.0060691, 0.0064877, 0.0060596, 0.0064914, -0.0004223, 0.0004281
6: -0.0014392, -0.0009014, -0.0014287, -0.0008955, -0.0004571, 0.0004385
7: -0.0083017, -0.0079822, -0.0083049, -0.0079754, -0.0003263, 0.0003227
8: 0.0049905, 0.0070261, 0.0049682, 0.0069865, -0.0015278, 0.0015844
9: -0.0036866, -0.0032869, -0.0036867, -0.0032745, -0.0004121, 0.0003999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.15 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002580, upper bound: 0.0002239
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002600, upper bound: 0.0002472
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009703, 0.0011012, 0.0009678, 0.0011009, -0.0001271, 0.0001298
1: 0.9936085, 0.9940149, 0.9936112, 0.9940214, -0.0003686, 0.0003605
2: -0.0066783, -0.0051290, -0.0066492, -0.0051178, -0.0013187, 0.0012720
3: 0.0038262, 0.0040218, 0.0038235, 0.0040199, -0.0001640, 0.0001678
4: 0.0024707, 0.0036951, 0.0024619, 0.0036722, -0.0010606, 0.0010974
5: 0.0060691, 0.0064877, 0.0060628, 0.0064901, -0.0004210, 0.0004249
6: -0.0014392, -0.0009014, -0.0014291, -0.0008975, -0.0004544, 0.0004382
7: -0.0083017, -0.0079822, -0.0083038, -0.0079777, -0.0003240, 0.0003216
8: 0.0049905, 0.0070261, 0.0049758, 0.0069878, -0.0015232, 0.0015739
9: -0.0036866, -0.0032869, -0.0036867, -0.0032787, -0.0004079, 0.0003998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.19 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002580, upper bound: 0.0002256
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002600, upper bound: 0.0002532
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009569, 0.0011006, 0.0009645, 0.0011009, -0.0001404, 0.0001325
1: 0.9936160, 0.9940488, 0.9936113, 0.9940295, -0.0003706, 0.0003932
2: -0.0065994, -0.0050702, -0.0066482, -0.0051035, -0.0012909, 0.0013354
3: 0.0038122, 0.0040166, 0.0038201, 0.0040198, -0.0001776, 0.0001669
4: 0.0024242, 0.0036328, 0.0024505, 0.0036714, -0.0011140, 0.0010840
5: 0.0060362, 0.0065004, 0.0060548, 0.0064932, -0.0004570, 0.0004456
6: -0.0014118, -0.0008810, -0.0014287, -0.0008926, -0.0004443, 0.0004600
7: -0.0083128, -0.0079585, -0.0083065, -0.0079719, -0.0003409, 0.0003480
8: 0.0049132, 0.0069224, 0.0049570, 0.0069865, -0.0016024, 0.0015185
9: -0.0036872, -0.0032439, -0.0036868, -0.0032682, -0.0004189, 0.0004429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003221
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003387
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009569, 0.0011006, 0.0009658, 0.0011009, -0.0001404, 0.0001313
1: 0.9936160, 0.9940488, 0.9936112, 0.9940263, -0.0003677, 0.0003932
2: -0.0065994, -0.0050702, -0.0066492, -0.0051090, -0.0012824, 0.0013358
3: 0.0038122, 0.0040166, 0.0038215, 0.0040199, -0.0001776, 0.0001657
4: 0.0024242, 0.0036328, 0.0024549, 0.0036722, -0.0011137, 0.0010768
5: 0.0060362, 0.0065004, 0.0060579, 0.0064920, -0.0004558, 0.0004425
6: -0.0014118, -0.0008810, -0.0014291, -0.0008945, -0.0004414, 0.0004602
7: -0.0083128, -0.0079585, -0.0083055, -0.0079741, -0.0003387, 0.0003469
8: 0.0049132, 0.0069224, 0.0049643, 0.0069878, -0.0015999, 0.0015078
9: -0.0036872, -0.0032439, -0.0036868, -0.0032723, -0.0004149, 0.0004429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003245
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003397
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009685, 0.0011012, 0.0009565, 0.0011005, -0.0001287, 0.0001413
1: 0.9936085, 0.9940197, 0.9936161, 0.9940498, -0.0003996, 0.0003609
2: -0.0066783, -0.0051207, -0.0065980, -0.0050682, -0.0013994, 0.0012613
3: 0.0038242, 0.0040218, 0.0038117, 0.0040165, -0.0001629, 0.0001814
4: 0.0024642, 0.0036952, 0.0024226, 0.0036317, -0.0010605, 0.0011637
5: 0.0060645, 0.0064895, 0.0060351, 0.0065008, -0.0004364, 0.0004544
6: -0.0014392, -0.0008985, -0.0014113, -0.0008803, -0.0004824, 0.0004341
7: -0.0083032, -0.0079788, -0.0083132, -0.0079577, -0.0003455, 0.0003343
8: 0.0049797, 0.0070261, 0.0049106, 0.0069206, -0.0014859, 0.0016848
9: -0.0036866, -0.0032808, -0.0036872, -0.0032425, -0.0004442, 0.0004064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002660, upper bound: 0.0002698
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.75 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003066, upper bound: 0.0002636
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003145, upper bound: 0.0002942
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009685, 0.0011012, 0.0009576, 0.0011006, -0.0001287, 0.0001402
1: 0.9936085, 0.9940197, 0.9936160, 0.9940471, -0.0003971, 0.0003609
2: -0.0066783, -0.0051207, -0.0065994, -0.0050730, -0.0013935, 0.0012618
3: 0.0038242, 0.0040218, 0.0038128, 0.0040166, -0.0001630, 0.0001805
4: 0.0024642, 0.0036952, 0.0024265, 0.0036328, -0.0010602, 0.0011581
5: 0.0060645, 0.0064895, 0.0060378, 0.0064998, -0.0004353, 0.0004517
6: -0.0014392, -0.0008985, -0.0014118, -0.0008820, -0.0004805, 0.0004343
7: -0.0083032, -0.0079788, -0.0083123, -0.0079597, -0.0003436, 0.0003334
8: 0.0049797, 0.0070261, 0.0049170, 0.0069224, -0.0014830, 0.0016784
9: -0.0036866, -0.0032808, -0.0036871, -0.0032460, -0.0004406, 0.0004063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002660, upper bound: 0.0002704
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.84 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003066, upper bound: 0.0002638
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003145, upper bound: 0.0002947
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009552, 0.0011006, 0.0009547, 0.0011005, -0.0001418, 0.0001426
1: 0.9936160, 0.9940532, 0.9936161, 0.9940546, -0.0003985, 0.0003937
2: -0.0065994, -0.0050624, -0.0065980, -0.0050602, -0.0013685, 0.0013275
3: 0.0038103, 0.0040166, 0.0038098, 0.0040165, -0.0001767, 0.0001793
4: 0.0024181, 0.0036328, 0.0024163, 0.0036317, -0.0011137, 0.0011443
5: 0.0060319, 0.0065021, 0.0060306, 0.0065026, -0.0004707, 0.0004715
6: -0.0014118, -0.0008783, -0.0014113, -0.0008775, -0.0004713, 0.0004570
7: -0.0083143, -0.0079554, -0.0083147, -0.0079545, -0.0003598, 0.0003593
8: 0.0049030, 0.0069224, 0.0049001, 0.0069206, -0.0015648, 0.0016216
9: -0.0036872, -0.0032383, -0.0036873, -0.0032367, -0.0004506, 0.0004490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003309
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003654
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009552, 0.0011006, 0.0009557, 0.0011006, -0.0001419, 0.0001416
1: 0.9936160, 0.9940532, 0.9936160, 0.9940521, -0.0003965, 0.0003938
2: -0.0065994, -0.0050624, -0.0065994, -0.0050646, -0.0013628, 0.0013279
3: 0.0038103, 0.0040166, 0.0038108, 0.0040166, -0.0001768, 0.0001787
4: 0.0024181, 0.0036328, 0.0024198, 0.0036328, -0.0011132, 0.0011381
5: 0.0060319, 0.0065021, 0.0060331, 0.0065016, -0.0004698, 0.0004690
6: -0.0014118, -0.0008783, -0.0014118, -0.0008791, -0.0004695, 0.0004571
7: -0.0083143, -0.0079554, -0.0083139, -0.0079563, -0.0003580, 0.0003585
8: 0.0049030, 0.0069224, 0.0049059, 0.0069224, -0.0015652, 0.0016186
9: -0.0036872, -0.0032383, -0.0036872, -0.0032398, -0.0004474, 0.0004490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003309
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003659
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.38 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002574, upper bound: 0.0002232
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002589, upper bound: 0.0002445
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002574, upper bound: 0.0002245
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002589, upper bound: 0.0002465
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003207, upper bound: 0.0003207
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003207, upper bound: 0.0003385
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003207, upper bound: 0.0003215
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003207, upper bound: 0.0003391
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003067, upper bound: 0.0002634
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003144, upper bound: 0.0002927
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003067, upper bound: 0.0002634
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003144, upper bound: 0.0002928
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002192, upper bound: 0.0002411
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003173, upper bound: 0.0003368
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002192, upper bound: 0.0002957
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003173, upper bound: 0.0003655
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002580, upper bound: 0.0002239
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002600, upper bound: 0.0002472
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002580, upper bound: 0.0002256
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0002600, upper bound: 0.0002532
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003221
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003387
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003245
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003397
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003066, upper bound: 0.0002636
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003145, upper bound: 0.0002942
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003066, upper bound: 0.0002638
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003145, upper bound: 0.0002947
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003309
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003654
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003309
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 1, lower bound: -0.0003215, upper bound: 0.0003659

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0009668, 0.0011009, 0.0009645, 0.0011009, -0.0001305, 0.0001329
1: 0.9936113, 0.9940238, 0.9936113, 0.9940295, -0.0003749, 0.0003681
2: -0.0066482, -0.0051137, -0.0066482, -0.0051035, -0.0013071, 0.0012825
3: 0.0038225, 0.0040198, 0.0038201, 0.0040198, -0.0001668, 0.0001699
4: 0.0024586, 0.0036714, 0.0024505, 0.0036714, -0.0010721, 0.0010877
5: 0.0060605, 0.0064910, 0.0060548, 0.0064932, -0.0004327, 0.0004362
6: -0.0014287, -0.0008961, -0.0014287, -0.0008926, -0.0004506, 0.0004416
7: -0.0083046, -0.0079760, -0.0083065, -0.0079719, -0.0003327, 0.0003305
8: 0.0049704, 0.0069865, 0.0049570, 0.0069865, -0.0015328, 0.0015743
9: -0.0036867, -0.0032757, -0.0036868, -0.0032682, -0.0004185, 0.0004112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.20 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002725, upper bound: 0.0002924
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002951, upper bound: 0.0002951
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0009570, 0.0011005, 0.0009645, 0.0011009, -0.0001403, 0.0001325
1: 0.9936161, 0.9940484, 0.9936113, 0.9940295, -0.0003704, 0.0003934
2: -0.0065980, -0.0050705, -0.0066482, -0.0051035, -0.0012819, 0.0013407
3: 0.0038123, 0.0040165, 0.0038201, 0.0040198, -0.0001778, 0.0001668
4: 0.0024245, 0.0036317, 0.0024505, 0.0036714, -0.0011181, 0.0010762
5: 0.0060364, 0.0065003, 0.0060548, 0.0064932, -0.0004568, 0.0004455
6: -0.0014113, -0.0008811, -0.0014287, -0.0008926, -0.0004413, 0.0004618
7: -0.0083127, -0.0079587, -0.0083065, -0.0079719, -0.0003408, 0.0003479
8: 0.0049137, 0.0069206, 0.0049570, 0.0069865, -0.0016093, 0.0015139
9: -0.0036872, -0.0032442, -0.0036868, -0.0032682, -0.0004189, 0.0004426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.25 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002725, upper bound: 0.0002924
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002951, upper bound: 0.0003120
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0009668, 0.0011009, 0.0009658, 0.0011009, -0.0001305, 0.0001316
1: 0.9936113, 0.9940238, 0.9936112, 0.9940263, -0.0003720, 0.0003680
2: -0.0066482, -0.0051137, -0.0066492, -0.0051090, -0.0012994, 0.0012848
3: 0.0038225, 0.0040198, 0.0038215, 0.0040199, -0.0001667, 0.0001688
4: 0.0024586, 0.0036714, 0.0024549, 0.0036722, -0.0010735, 0.0010816
5: 0.0060605, 0.0064910, 0.0060579, 0.0064920, -0.0004315, 0.0004331
6: -0.0014287, -0.0008961, -0.0014291, -0.0008945, -0.0004480, 0.0004425
7: -0.0083046, -0.0079760, -0.0083055, -0.0079741, -0.0003304, 0.0003294
8: 0.0049704, 0.0069865, 0.0049643, 0.0069878, -0.0015308, 0.0015641
9: -0.0036867, -0.0032757, -0.0036868, -0.0032723, -0.0004144, 0.0004111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.23 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002729, upper bound: 0.0002931
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002974, upper bound: 0.0002958
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0009570, 0.0011005, 0.0009658, 0.0011009, -0.0001403, 0.0001313
1: 0.9936161, 0.9940484, 0.9936112, 0.9940263, -0.0003675, 0.0003933
2: -0.0065980, -0.0050705, -0.0066492, -0.0051090, -0.0012741, 0.0013430
3: 0.0038123, 0.0040165, 0.0038215, 0.0040199, -0.0001777, 0.0001657
4: 0.0024245, 0.0036317, 0.0024549, 0.0036722, -0.0011195, 0.0010700
5: 0.0060364, 0.0065003, 0.0060579, 0.0064920, -0.0004556, 0.0004424
6: -0.0014113, -0.0008811, -0.0014291, -0.0008945, -0.0004386, 0.0004627
7: -0.0083127, -0.0079587, -0.0083055, -0.0079741, -0.0003386, 0.0003468
8: 0.0049137, 0.0069206, 0.0049643, 0.0069878, -0.0016073, 0.0015037
9: -0.0036872, -0.0032442, -0.0036868, -0.0032723, -0.0004149, 0.0004426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.28 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002729, upper bound: 0.0003082
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002974, upper bound: 0.0002958
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010059, 0.0011008, 0.0009569, 0.0011005, -0.0000911, 0.0001399
1: 0.9936125, 0.9939255, 0.9936161, 0.9940489, -0.0003862, 0.0002642
2: -0.0066365, -0.0052848, -0.0065980, -0.0050700, -0.0012390, 0.0010577
3: 0.0038633, 0.0040191, 0.0038121, 0.0040165, -0.0001219, 0.0001723
4: 0.0025938, 0.0036621, 0.0024241, 0.0036317, -0.0008994, 0.0010248
5: 0.0061561, 0.0064540, 0.0060361, 0.0065005, -0.0003443, 0.0004179
6: -0.0014247, -0.0009555, -0.0014113, -0.0008809, -0.0004274, 0.0003634
7: -0.0082722, -0.0080166, -0.0083128, -0.0079584, -0.0003138, 0.0002962
8: 0.0051952, 0.0069711, 0.0049130, 0.0069206, -0.0012189, 0.0015065
9: -0.0036850, -0.0034005, -0.0036872, -0.0032438, -0.0004412, 0.0002867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002836, upper bound: 0.0002585
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002836, upper bound: 0.0002585
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0009859, 0.0011011, 0.0009565, 0.0011005, -0.0001100, 0.0001413
1: 0.9936086, 0.9939758, 0.9936161, 0.9940498, -0.0003997, 0.0003023
2: -0.0066771, -0.0051973, -0.0065980, -0.0050682, -0.0013510, 0.0010576
3: 0.0038425, 0.0040217, 0.0038117, 0.0040165, -0.0001344, 0.0001815
4: 0.0025247, 0.0036942, 0.0024226, 0.0036317, -0.0009165, 0.0011052
5: 0.0061072, 0.0064729, 0.0060351, 0.0065008, -0.0003936, 0.0004378
6: -0.0014388, -0.0009251, -0.0014113, -0.0008803, -0.0004669, 0.0003624
7: -0.0082888, -0.0080090, -0.0083132, -0.0079577, -0.0003311, 0.0003042
8: 0.0050802, 0.0070245, 0.0049106, 0.0069206, -0.0011714, 0.0016801
9: -0.0036859, -0.0033367, -0.0036872, -0.0032425, -0.0004434, 0.0003505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002877, upper bound: 0.0002877
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002877, upper bound: 0.0002877
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010059, 0.0011008, 0.0009578, 0.0011006, -0.0000911, 0.0001389
1: 0.9936125, 0.9939255, 0.9936160, 0.9940464, -0.0003840, 0.0002642
2: -0.0066365, -0.0052848, -0.0065994, -0.0050741, -0.0012313, 0.0010628
3: 0.0038633, 0.0040191, 0.0038131, 0.0040166, -0.0001219, 0.0001716
4: 0.0025938, 0.0036621, 0.0024273, 0.0036328, -0.0009030, 0.0010186
5: 0.0061561, 0.0064540, 0.0060384, 0.0064996, -0.0003434, 0.0004156
6: -0.0014247, -0.0009555, -0.0014118, -0.0008824, -0.0004247, 0.0003652
7: -0.0082722, -0.0080166, -0.0083121, -0.0079601, -0.0003121, 0.0002954
8: 0.0051952, 0.0069711, 0.0049184, 0.0069224, -0.0012206, 0.0014964
9: -0.0036850, -0.0034005, -0.0036871, -0.0032468, -0.0004381, 0.0002866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002851, upper bound: 0.0002585
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002851, upper bound: 0.0002634
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0009859, 0.0011011, 0.0009576, 0.0011006, -0.0001101, 0.0001402
1: 0.9936086, 0.9939758, 0.9936160, 0.9940471, -0.0003971, 0.0003033
2: -0.0066771, -0.0051973, -0.0065994, -0.0050730, -0.0013423, 0.0010681
3: 0.0038425, 0.0040217, 0.0038128, 0.0040166, -0.0001351, 0.0001806
4: 0.0025247, 0.0036942, 0.0024265, 0.0036328, -0.0009246, 0.0010983
5: 0.0061072, 0.0064729, 0.0060378, 0.0064998, -0.0003926, 0.0004351
6: -0.0014388, -0.0009251, -0.0014118, -0.0008820, -0.0004639, 0.0003659
7: -0.0082888, -0.0080090, -0.0083123, -0.0079597, -0.0003291, 0.0003033
8: 0.0050802, 0.0070245, 0.0049170, 0.0069224, -0.0011844, 0.0016687
9: -0.0036859, -0.0033367, -0.0036871, -0.0032460, -0.0004398, 0.0003505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002895, upper bound: 0.0002879
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002895, upper bound: 0.0002928
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0009542, 0.0011005, 0.0009708, 0.0011013, -0.0001435, 0.0001266
1: 0.9936163, 0.9940557, 0.9936066, 0.9940138, -0.0003581, 0.0004042
2: -0.0065967, -0.0050580, -0.0066977, -0.0051309, -0.0012877, 0.0014125
3: 0.0038093, 0.0040164, 0.0038266, 0.0040231, -0.0001833, 0.0001626
4: 0.0024146, 0.0036307, 0.0024722, 0.0037105, -0.0011711, 0.0010783
5: 0.0060294, 0.0065030, 0.0060701, 0.0064873, -0.0004579, 0.0004329
6: -0.0014109, -0.0008768, -0.0014459, -0.0009021, -0.0004434, 0.0004870
7: -0.0083151, -0.0079536, -0.0083013, -0.0079829, -0.0003322, 0.0003477
8: 0.0048972, 0.0069189, 0.0049930, 0.0070516, -0.0016998, 0.0015188
9: -0.0036873, -0.0032350, -0.0036865, -0.0032882, -0.0003991, 0.0004515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003163, upper bound: 0.0003366
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003163, upper bound: 0.0003368
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0009448, 0.0011001, 0.0009576, 0.0011007, -0.0001523, 0.0001393
1: 0.9936219, 0.9940792, 0.9936143, 0.9940470, -0.0003857, 0.0004206
2: -0.0065369, -0.0050170, -0.0066170, -0.0050729, -0.0012935, 0.0013912
3: 0.0037995, 0.0040125, 0.0038128, 0.0040178, -0.0001882, 0.0001726
4: 0.0023822, 0.0035834, 0.0024264, 0.0036467, -0.0011617, 0.0010922
5: 0.0060065, 0.0065119, 0.0060377, 0.0064998, -0.0004933, 0.0004742
6: -0.0013901, -0.0008625, -0.0014179, -0.0008819, -0.0004449, 0.0004791
7: -0.0083229, -0.0079371, -0.0083123, -0.0079596, -0.0003633, 0.0003752
8: 0.0048434, 0.0068403, 0.0049168, 0.0069455, -0.0016517, 0.0015114
9: -0.0036877, -0.0032051, -0.0036871, -0.0032459, -0.0004418, 0.0004820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003072, upper bound: 0.0002955
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003072, upper bound: 0.0002957
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0009542, 0.0011005, 0.0009575, 0.0011007, -0.0001430, 0.0001397
1: 0.9936163, 0.9940557, 0.9936141, 0.9940473, -0.0003899, 0.0003980
2: -0.0065967, -0.0050580, -0.0066194, -0.0050725, -0.0013391, 0.0013475
3: 0.0038093, 0.0040164, 0.0038127, 0.0040179, -0.0001789, 0.0001754
4: 0.0024146, 0.0036307, 0.0024261, 0.0036486, -0.0011272, 0.0011224
5: 0.0060294, 0.0065030, 0.0060375, 0.0064999, -0.0004705, 0.0004655
6: -0.0014109, -0.0008768, -0.0014187, -0.0008818, -0.0004611, 0.0004640
7: -0.0083151, -0.0079536, -0.0083124, -0.0079595, -0.0003556, 0.0003587
8: 0.0048972, 0.0069189, 0.0049163, 0.0069487, -0.0015941, 0.0015819
9: -0.0036873, -0.0032350, -0.0036871, -0.0032457, -0.0004416, 0.0004521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003654, upper bound: 0.0003654
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003654, upper bound: 0.0003655
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0009680, 0.0011009, 0.0009645, 0.0011009, -0.0001294, 0.0001328
1: 0.9936112, 0.9940209, 0.9936113, 0.9940295, -0.0003745, 0.0003652
2: -0.0066492, -0.0051186, -0.0066482, -0.0051035, -0.0013114, 0.0012760
3: 0.0038237, 0.0040199, 0.0038201, 0.0040198, -0.0001655, 0.0001696
4: 0.0024625, 0.0036722, 0.0024505, 0.0036714, -0.0010670, 0.0010913
5: 0.0060633, 0.0064899, 0.0060548, 0.0064932, -0.0004299, 0.0004351
6: -0.0014291, -0.0008978, -0.0014287, -0.0008926, -0.0004520, 0.0004394
7: -0.0083036, -0.0079780, -0.0083065, -0.0079719, -0.0003317, 0.0003285
8: 0.0049769, 0.0069878, 0.0049570, 0.0069865, -0.0015244, 0.0015690
9: -0.0036867, -0.0032793, -0.0036868, -0.0032682, -0.0004184, 0.0004075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.21 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002943
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0002974
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0009580, 0.0011006, 0.0009645, 0.0011009, -0.0001393, 0.0001325
1: 0.9936160, 0.9940461, 0.9936113, 0.9940295, -0.0003705, 0.0003911
2: -0.0065994, -0.0050747, -0.0066482, -0.0051035, -0.0012869, 0.0013393
3: 0.0038133, 0.0040166, 0.0038201, 0.0040198, -0.0001769, 0.0001668
4: 0.0024278, 0.0036328, 0.0024505, 0.0036714, -0.0011170, 0.0010797
5: 0.0060388, 0.0064994, 0.0060548, 0.0064932, -0.0004544, 0.0004446
6: -0.0014118, -0.0008826, -0.0014287, -0.0008926, -0.0004430, 0.0004613
7: -0.0083119, -0.0079604, -0.0083065, -0.0079719, -0.0003400, 0.0003462
8: 0.0049192, 0.0069224, 0.0049570, 0.0069865, -0.0016075, 0.0015156
9: -0.0036871, -0.0032473, -0.0036868, -0.0032682, -0.0004189, 0.0004396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.29 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0003078
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0003128
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0009680, 0.0011009, 0.0009658, 0.0011009, -0.0001294, 0.0001316
1: 0.9936112, 0.9940209, 0.9936112, 0.9940263, -0.0003720, 0.0003653
2: -0.0066492, -0.0051186, -0.0066492, -0.0051090, -0.0013028, 0.0012791
3: 0.0038237, 0.0040199, 0.0038215, 0.0040199, -0.0001656, 0.0001687
4: 0.0024625, 0.0036722, 0.0024549, 0.0036722, -0.0010689, 0.0010845
5: 0.0060633, 0.0064899, 0.0060579, 0.0064920, -0.0004287, 0.0004320
6: -0.0014291, -0.0008978, -0.0014291, -0.0008945, -0.0004490, 0.0004405
7: -0.0083036, -0.0079780, -0.0083055, -0.0079741, -0.0003295, 0.0003274
8: 0.0049769, 0.0069878, 0.0049643, 0.0069878, -0.0015255, 0.0015643
9: -0.0036867, -0.0032793, -0.0036868, -0.0032723, -0.0004144, 0.0004075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.22 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002975
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0003008
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0009580, 0.0011006, 0.0009658, 0.0011009, -0.0001394, 0.0001313
1: 0.9936160, 0.9940461, 0.9936112, 0.9940263, -0.0003676, 0.0003910
2: -0.0065994, -0.0050747, -0.0066492, -0.0051090, -0.0012784, 0.0013392
3: 0.0038133, 0.0040166, 0.0038215, 0.0040199, -0.0001768, 0.0001657
4: 0.0024278, 0.0036328, 0.0024549, 0.0036722, -0.0011163, 0.0010728
5: 0.0060388, 0.0064994, 0.0060579, 0.0064920, -0.0004532, 0.0004415
6: -0.0014118, -0.0008826, -0.0014291, -0.0008945, -0.0004401, 0.0004614
7: -0.0083119, -0.0079604, -0.0083055, -0.0079741, -0.0003378, 0.0003451
8: 0.0049192, 0.0069224, 0.0049643, 0.0069878, -0.0016043, 0.0015056
9: -0.0036871, -0.0032473, -0.0036868, -0.0032723, -0.0004148, 0.0004395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.15 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0003095
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0003153
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0010068, 0.0011009, 0.0009566, 0.0011005, -0.0000901, 0.0001402
1: 0.9936122, 0.9939228, 0.9936161, 0.9940495, -0.0003874, 0.0002620
2: -0.0066393, -0.0052890, -0.0065980, -0.0050689, -0.0012427, 0.0010625
3: 0.0038644, 0.0040192, 0.0038119, 0.0040165, -0.0001212, 0.0001730
4: 0.0025972, 0.0036643, 0.0024232, 0.0036317, -0.0009032, 0.0010272
5: 0.0061585, 0.0064531, 0.0060355, 0.0065007, -0.0003422, 0.0004176
6: -0.0014256, -0.0009570, -0.0014113, -0.0008806, -0.0004287, 0.0003651
7: -0.0082714, -0.0080161, -0.0083130, -0.0079580, -0.0003134, 0.0002969
8: 0.0052008, 0.0069748, 0.0049116, 0.0069206, -0.0012249, 0.0015157
9: -0.0036849, -0.0034036, -0.0036872, -0.0032430, -0.0004419, 0.0002836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002585
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002636
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0009875, 0.0011012, 0.0009565, 0.0011005, -0.0001085, 0.0001413
1: 0.9936085, 0.9939718, 0.9936161, 0.9940498, -0.0003996, 0.0002985
2: -0.0066783, -0.0052042, -0.0065980, -0.0050682, -0.0013488, 0.0010678
3: 0.0038441, 0.0040218, 0.0038117, 0.0040165, -0.0001329, 0.0001814
4: 0.0025301, 0.0036952, 0.0024226, 0.0036317, -0.0009246, 0.0011049
5: 0.0061111, 0.0064715, 0.0060351, 0.0065008, -0.0003898, 0.0004363
6: -0.0014392, -0.0009275, -0.0014113, -0.0008803, -0.0004660, 0.0003659
7: -0.0082875, -0.0080088, -0.0083132, -0.0079577, -0.0003298, 0.0003044
8: 0.0050893, 0.0070261, 0.0049106, 0.0069206, -0.0011848, 0.0016776
9: -0.0036858, -0.0033417, -0.0036872, -0.0032425, -0.0004433, 0.0003455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002879, upper bound: 0.0002895
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002879, upper bound: 0.0002942
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0010068, 0.0011009, 0.0009579, 0.0011006, -0.0000901, 0.0001389
1: 0.9936122, 0.9939228, 0.9936160, 0.9940464, -0.0003840, 0.0002619
2: -0.0066393, -0.0052890, -0.0065994, -0.0050745, -0.0012345, 0.0010590
3: 0.0038644, 0.0040192, 0.0038132, 0.0040166, -0.0001211, 0.0001717
4: 0.0025972, 0.0036643, 0.0024276, 0.0036328, -0.0008998, 0.0010206
5: 0.0061585, 0.0064531, 0.0060386, 0.0064995, -0.0003410, 0.0004145
6: -0.0014256, -0.0009570, -0.0014118, -0.0008825, -0.0004259, 0.0003639
7: -0.0082714, -0.0080161, -0.0083120, -0.0079603, -0.0003112, 0.0002959
8: 0.0052008, 0.0069748, 0.0049189, 0.0069224, -0.0012169, 0.0015018
9: -0.0036849, -0.0034036, -0.0036871, -0.0032471, -0.0004378, 0.0002835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002854, upper bound: 0.0002585
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002854, upper bound: 0.0002638
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0009875, 0.0011012, 0.0009576, 0.0011006, -0.0001085, 0.0001402
1: 0.9936085, 0.9939718, 0.9936160, 0.9940471, -0.0003971, 0.0002988
2: -0.0066783, -0.0052042, -0.0065994, -0.0050730, -0.0013439, 0.0010563
3: 0.0038441, 0.0040218, 0.0038128, 0.0040166, -0.0001332, 0.0001805
4: 0.0025301, 0.0036952, 0.0024265, 0.0036328, -0.0009151, 0.0010993
5: 0.0061111, 0.0064715, 0.0060378, 0.0064998, -0.0003887, 0.0004336
6: -0.0014392, -0.0009275, -0.0014118, -0.0008820, -0.0004644, 0.0003620
7: -0.0082875, -0.0080088, -0.0083123, -0.0079597, -0.0003278, 0.0003035
8: 0.0050893, 0.0070261, 0.0049170, 0.0069224, -0.0011720, 0.0016711
9: -0.0036858, -0.0033417, -0.0036871, -0.0032460, -0.0004398, 0.0003454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002900, upper bound: 0.0002901
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002900, upper bound: 0.0002947
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0009680, 0.0011009, 0.0009547, 0.0011005, -0.0001290, 0.0001427
1: 0.9936112, 0.9940209, 0.9936161, 0.9940546, -0.0003995, 0.0003610
2: -0.0066492, -0.0051186, -0.0065980, -0.0050602, -0.0013671, 0.0012541
3: 0.0038237, 0.0040199, 0.0038098, 0.0040165, -0.0001626, 0.0001804
4: 0.0024625, 0.0036722, 0.0024163, 0.0036317, -0.0010557, 0.0011353
5: 0.0060633, 0.0064899, 0.0060306, 0.0065026, -0.0004393, 0.0004593
6: -0.0014291, -0.0008978, -0.0014113, -0.0008775, -0.0004713, 0.0004315
7: -0.0083036, -0.0079780, -0.0083147, -0.0079545, -0.0003492, 0.0003367
8: 0.0049769, 0.0069878, 0.0049001, 0.0069206, -0.0014684, 0.0016422
9: -0.0036867, -0.0032793, -0.0036873, -0.0032367, -0.0004500, 0.0004080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.22 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0003053
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0003065
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0009580, 0.0011006, 0.0009547, 0.0011005, -0.0001392, 0.0001426
1: 0.9936160, 0.9940461, 0.9936161, 0.9940546, -0.0003980, 0.0003890
2: -0.0065994, -0.0050747, -0.0065980, -0.0050602, -0.0013605, 0.0013218
3: 0.0038133, 0.0040166, 0.0038098, 0.0040165, -0.0001752, 0.0001790
4: 0.0024278, 0.0036328, 0.0024163, 0.0036317, -0.0011092, 0.0011355
5: 0.0060388, 0.0064994, 0.0060306, 0.0065026, -0.0004638, 0.0004688
6: -0.0014118, -0.0008826, -0.0014113, -0.0008775, -0.0004688, 0.0004552
7: -0.0083119, -0.0079604, -0.0083147, -0.0079545, -0.0003574, 0.0003543
8: 0.0049192, 0.0069224, 0.0049001, 0.0069206, -0.0015679, 0.0016150
9: -0.0036871, -0.0032473, -0.0036873, -0.0032367, -0.0004505, 0.0004400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.30 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002943
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0003396
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0009680, 0.0011009, 0.0009557, 0.0011006, -0.0001290, 0.0001417
1: 0.9936112, 0.9940209, 0.9936160, 0.9940521, -0.0003976, 0.0003612
2: -0.0066492, -0.0051186, -0.0065994, -0.0050646, -0.0013624, 0.0012573
3: 0.0038237, 0.0040199, 0.0038108, 0.0040166, -0.0001627, 0.0001798
4: 0.0024625, 0.0036722, 0.0024198, 0.0036328, -0.0010574, 0.0011315
5: 0.0060633, 0.0064899, 0.0060331, 0.0065016, -0.0004384, 0.0004569
6: -0.0014291, -0.0008978, -0.0014118, -0.0008791, -0.0004696, 0.0004326
7: -0.0083036, -0.0079780, -0.0083139, -0.0079563, -0.0003474, 0.0003359
8: 0.0049769, 0.0069878, 0.0049059, 0.0069224, -0.0014724, 0.0016426
9: -0.0036867, -0.0032793, -0.0036872, -0.0032398, -0.0004468, 0.0004079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.26 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0003073
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0003084
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0009580, 0.0011006, 0.0009557, 0.0011006, -0.0001392, 0.0001416
1: 0.9936160, 0.9940461, 0.9936160, 0.9940521, -0.0003961, 0.0003891
2: -0.0065994, -0.0050747, -0.0065994, -0.0050646, -0.0013551, 0.0013242
3: 0.0038133, 0.0040166, 0.0038108, 0.0040166, -0.0001752, 0.0001784
4: 0.0024278, 0.0036328, 0.0024198, 0.0036328, -0.0011118, 0.0011300
5: 0.0060388, 0.0064994, 0.0060331, 0.0065016, -0.0004629, 0.0004663
6: -0.0014118, -0.0008826, -0.0014118, -0.0008791, -0.0004669, 0.0004560
7: -0.0083119, -0.0079604, -0.0083139, -0.0079563, -0.0003557, 0.0003535
8: 0.0049192, 0.0069224, 0.0049059, 0.0069224, -0.0015701, 0.0016127
9: -0.0036871, -0.0032473, -0.0036872, -0.0032398, -0.0004473, 0.0004400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 136
type: B, layer: 3, pos: 81

Time for candidate selection: 7.25 seconds

### Candidate
type: B, layer: 3, pos: 73

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002975
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002942
time: 1.26 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 10.28 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002725, upper bound: 0.0002924
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002951, upper bound: 0.0002951
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002725, upper bound: 0.0002924
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002951, upper bound: 0.0003120
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002729, upper bound: 0.0002931
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002974, upper bound: 0.0002958
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002729, upper bound: 0.0003082
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002974, upper bound: 0.0002958
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002836, upper bound: 0.0002585
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002836, upper bound: 0.0002585
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002877, upper bound: 0.0002877
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002877, upper bound: 0.0002877
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002851, upper bound: 0.0002585
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002851, upper bound: 0.0002634
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002895, upper bound: 0.0002879
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002895, upper bound: 0.0002928
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003163, upper bound: 0.0003366
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003163, upper bound: 0.0003368
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003072, upper bound: 0.0002955
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003072, upper bound: 0.0002957
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003654, upper bound: 0.0003654
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003654, upper bound: 0.0003655
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002943
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0002974
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0003078
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0003128
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002975
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0003008
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0003095
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0003153
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002585
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002835, upper bound: 0.0002636
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002879, upper bound: 0.0002895
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002879, upper bound: 0.0002942
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002854, upper bound: 0.0002585
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002854, upper bound: 0.0002638
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002900, upper bound: 0.0002901
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002900, upper bound: 0.0002947
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0003053
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0003065
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002943
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0003396
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0003073
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0003084
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002975
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.28
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002942

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009686, 0.0011009, 0.0010038, 0.0011008, -0.0001278, 0.0000935
1: 0.9936113, 0.9940193, 0.9936131, 0.9939306, -0.0002746, 0.0003523
2: -0.0066482, -0.0051214, -0.0066298, -0.0052758, -0.0011021, 0.0011189
3: 0.0038244, 0.0040198, 0.0038612, 0.0040186, -0.0001570, 0.0001278
4: 0.0024647, 0.0036714, 0.0025868, 0.0036568, -0.0009307, 0.0009249
5: 0.0060648, 0.0064893, 0.0061511, 0.0064560, -0.0003911, 0.0003382
6: -0.0014287, -0.0008988, -0.0014224, -0.0009524, -0.0003795, 0.0003858
7: -0.0083031, -0.0079791, -0.0082739, -0.0080179, -0.0002852, 0.0002948
8: 0.0049805, 0.0069865, 0.0051834, 0.0069623, -0.0013562, 0.0013067
9: -0.0036866, -0.0032813, -0.0036851, -0.0033940, -0.0002926, 0.0004037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.57 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002725, upper bound: 0.0002725
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002725, upper bound: 0.0002924
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009668, 0.0011009, 0.0009834, 0.0011009, -0.0001305, 0.0001130
1: 0.9936113, 0.9940238, 0.9936113, 0.9939823, -0.0003137, 0.0003681
2: -0.0066482, -0.0051137, -0.0066482, -0.0051860, -0.0011185, 0.0012320
3: 0.0038225, 0.0040198, 0.0038398, 0.0040198, -0.0001668, 0.0001407
4: 0.0024586, 0.0036714, 0.0025158, 0.0036714, -0.0010144, 0.0009538
5: 0.0060605, 0.0064910, 0.0061009, 0.0064754, -0.0004149, 0.0003901
6: -0.0014287, -0.0008961, -0.0014287, -0.0009212, -0.0003841, 0.0004253
7: -0.0083046, -0.0079760, -0.0082909, -0.0080051, -0.0002995, 0.0003149
8: 0.0049704, 0.0069865, 0.0050654, 0.0069865, -0.0015257, 0.0012785
9: -0.0036867, -0.0032757, -0.0036860, -0.0033285, -0.0003582, 0.0004103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.59 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002924, upper bound: 0.0002725
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002924, upper bound: 0.0002951
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009586, 0.0011005, 0.0010038, 0.0011008, -0.0001379, 0.0000932
1: 0.9936161, 0.9940445, 0.9936131, 0.9939306, -0.0002701, 0.0003781
2: -0.0065980, -0.0050774, -0.0066298, -0.0052758, -0.0010770, 0.0011789
3: 0.0038139, 0.0040165, 0.0038612, 0.0040186, -0.0001682, 0.0001246
4: 0.0024299, 0.0036317, 0.0025868, 0.0036568, -0.0009781, 0.0009134
5: 0.0060402, 0.0064988, 0.0061511, 0.0064560, -0.0004157, 0.0003477
6: -0.0014113, -0.0008835, -0.0014224, -0.0009524, -0.0003702, 0.0004066
7: -0.0083114, -0.0079614, -0.0082739, -0.0080179, -0.0002935, 0.0003125
8: 0.0049227, 0.0069206, 0.0051834, 0.0069623, -0.0014350, 0.0012464
9: -0.0036871, -0.0032492, -0.0036851, -0.0033940, -0.0002931, 0.0004358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002281, upper bound: 0.0002266
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002787, upper bound: 0.0003070
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009570, 0.0011005, 0.0009834, 0.0011009, -0.0001403, 0.0001127
1: 0.9936161, 0.9940484, 0.9936113, 0.9939823, -0.0003096, 0.0003934
2: -0.0065980, -0.0050705, -0.0066482, -0.0051860, -0.0010961, 0.0012902
3: 0.0038123, 0.0040165, 0.0038398, 0.0040198, -0.0001778, 0.0001378
4: 0.0024245, 0.0036317, 0.0025158, 0.0036714, -0.0010604, 0.0009461
5: 0.0060364, 0.0065003, 0.0061009, 0.0064754, -0.0004390, 0.0003994
6: -0.0014113, -0.0008811, -0.0014287, -0.0009212, -0.0003758, 0.0004455
7: -0.0083127, -0.0079587, -0.0082909, -0.0080051, -0.0003076, 0.0003322
8: 0.0049137, 0.0069206, 0.0050654, 0.0069865, -0.0016021, 0.0012242
9: -0.0036872, -0.0032442, -0.0036860, -0.0033285, -0.0003587, 0.0004418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002564, upper bound: 0.0002358
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003032, upper bound: 0.0003117
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009683, 0.0011009, 0.0010044, 0.0011008, -0.0001281, 0.0000930
1: 0.9936113, 0.9940198, 0.9936129, 0.9939291, -0.0002733, 0.0003536
2: -0.0066482, -0.0051202, -0.0066324, -0.0052783, -0.0010991, 0.0011261
3: 0.0038241, 0.0040198, 0.0038618, 0.0040188, -0.0001577, 0.0001273
4: 0.0024637, 0.0036714, 0.0025887, 0.0036589, -0.0009371, 0.0009227
5: 0.0060641, 0.0064896, 0.0061525, 0.0064554, -0.0003913, 0.0003371
6: -0.0014287, -0.0008983, -0.0014233, -0.0009533, -0.0003785, 0.0003882
7: -0.0083034, -0.0079786, -0.0082734, -0.0080174, -0.0002859, 0.0002948
8: 0.0049789, 0.0069865, 0.0051867, 0.0069659, -0.0013653, 0.0013025
9: -0.0036867, -0.0032804, -0.0036850, -0.0033958, -0.0002908, 0.0004046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.56 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002729, upper bound: 0.0002734
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002729, upper bound: 0.0002931
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009668, 0.0011009, 0.0009850, 0.0011009, -0.0001305, 0.0001114
1: 0.9936113, 0.9940238, 0.9936112, 0.9939779, -0.0003103, 0.0003680
2: -0.0066482, -0.0051137, -0.0066492, -0.0051931, -0.0011184, 0.0012345
3: 0.0038225, 0.0040198, 0.0038415, 0.0040199, -0.0001667, 0.0001394
4: 0.0024586, 0.0036714, 0.0025214, 0.0036722, -0.0010156, 0.0009537
5: 0.0060605, 0.0064910, 0.0061049, 0.0064738, -0.0004133, 0.0003861
6: -0.0014287, -0.0008961, -0.0014291, -0.0009237, -0.0003840, 0.0004262
7: -0.0083046, -0.0079760, -0.0082896, -0.0080080, -0.0002966, 0.0003136
8: 0.0049704, 0.0069865, 0.0050747, 0.0069878, -0.0015237, 0.0012783
9: -0.0036867, -0.0032757, -0.0036859, -0.0033336, -0.0003531, 0.0004102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.53 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002943, upper bound: 0.0002734
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002943, upper bound: 0.0002958
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009583, 0.0011005, 0.0010044, 0.0011008, -0.0001382, 0.0000926
1: 0.9936161, 0.9940454, 0.9936129, 0.9939291, -0.0002688, 0.0003794
2: -0.0065980, -0.0050761, -0.0066324, -0.0052783, -0.0010740, 0.0011860
3: 0.0038136, 0.0040165, 0.0038618, 0.0040188, -0.0001689, 0.0001241
4: 0.0024289, 0.0036317, 0.0025887, 0.0036589, -0.0009845, 0.0009112
5: 0.0060395, 0.0064991, 0.0061525, 0.0064554, -0.0004159, 0.0003466
6: -0.0014113, -0.0008831, -0.0014233, -0.0009533, -0.0003691, 0.0004090
7: -0.0083117, -0.0079609, -0.0082734, -0.0080174, -0.0002943, 0.0003125
8: 0.0049211, 0.0069206, 0.0051867, 0.0069659, -0.0014440, 0.0012422
9: -0.0036871, -0.0032483, -0.0036850, -0.0033958, -0.0002913, 0.0004367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002281, upper bound: 0.0002266
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002791, upper bound: 0.0003078
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009570, 0.0011005, 0.0009850, 0.0011009, -0.0001403, 0.0001111
1: 0.9936161, 0.9940484, 0.9936112, 0.9939779, -0.0003061, 0.0003933
2: -0.0065980, -0.0050705, -0.0066492, -0.0051931, -0.0010959, 0.0012927
3: 0.0038123, 0.0040165, 0.0038415, 0.0040199, -0.0001777, 0.0001365
4: 0.0024245, 0.0036317, 0.0025214, 0.0036722, -0.0010616, 0.0009460
5: 0.0060364, 0.0065003, 0.0061049, 0.0064738, -0.0004374, 0.0003954
6: -0.0014113, -0.0008811, -0.0014291, -0.0009237, -0.0003758, 0.0004464
7: -0.0083127, -0.0079587, -0.0082896, -0.0080080, -0.0003048, 0.0003309
8: 0.0049137, 0.0069206, 0.0050747, 0.0069878, -0.0016001, 0.0012240
9: -0.0036872, -0.0032442, -0.0036859, -0.0033336, -0.0003535, 0.0004417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002589, upper bound: 0.0002373
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003062, upper bound: 0.0003129
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010059, 0.0011008, 0.0009704, 0.0011011, -0.0000915, 0.0001264
1: 0.9936125, 0.9939255, 0.9936086, 0.9940147, -0.0003522, 0.0002701
2: -0.0066365, -0.0052848, -0.0066771, -0.0051292, -0.0011617, 0.0011072
3: 0.0038633, 0.0040191, 0.0038262, 0.0040217, -0.0001260, 0.0001583
4: 0.0025938, 0.0036621, 0.0024708, 0.0036942, -0.0009303, 0.0009637
5: 0.0061561, 0.0064540, 0.0060692, 0.0064877, -0.0003315, 0.0003849
6: -0.0014247, -0.0009555, -0.0014388, -0.0009015, -0.0004006, 0.0003810
7: -0.0082722, -0.0080166, -0.0083016, -0.0079822, -0.0002900, 0.0002850
8: 0.0051952, 0.0069711, 0.0049907, 0.0070245, -0.0012988, 0.0014050
9: -0.0036850, -0.0034005, -0.0036866, -0.0032870, -0.0003980, 0.0002861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.29 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002632, upper bound: 0.0002349
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002650, upper bound: 0.0002408
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010059, 0.0011008, 0.0009574, 0.0011005, -0.0000911, 0.0001394
1: 0.9936125, 0.9939255, 0.9936161, 0.9940475, -0.0003852, 0.0002642
2: -0.0066365, -0.0052848, -0.0065980, -0.0050723, -0.0012449, 0.0010570
3: 0.0038633, 0.0040191, 0.0038127, 0.0040165, -0.0001219, 0.0001721
4: 0.0025938, 0.0036621, 0.0024259, 0.0036317, -0.0008982, 0.0010294
5: 0.0061561, 0.0064540, 0.0060374, 0.0065000, -0.0003438, 0.0004166
6: -0.0014247, -0.0009555, -0.0014113, -0.0008817, -0.0004295, 0.0003632
7: -0.0082722, -0.0080166, -0.0083124, -0.0079594, -0.0003129, 0.0002958
8: 0.0051952, 0.0069711, 0.0049160, 0.0069206, -0.0012189, 0.0015143
9: -0.0036850, -0.0034005, -0.0036871, -0.0032455, -0.0004395, 0.0002866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.38 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002632, upper bound: 0.0002349
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002650, upper bound: 0.0002458
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009859, 0.0011011, 0.0009699, 0.0011011, -0.0001103, 0.0001279
1: 0.9936086, 0.9939758, 0.9936086, 0.9940161, -0.0003658, 0.0003056
2: -0.0066771, -0.0051973, -0.0066771, -0.0051271, -0.0012740, 0.0010767
3: 0.0038425, 0.0040217, 0.0038257, 0.0040217, -0.0001367, 0.0001675
4: 0.0025247, 0.0036942, 0.0024692, 0.0036942, -0.0009255, 0.0010444
5: 0.0061072, 0.0064729, 0.0060680, 0.0064881, -0.0003809, 0.0004049
6: -0.0014388, -0.0009251, -0.0014388, -0.0009008, -0.0004402, 0.0003693
7: -0.0082888, -0.0080090, -0.0083020, -0.0079814, -0.0003074, 0.0002931
8: 0.0050802, 0.0070245, 0.0049880, 0.0070245, -0.0012141, 0.0015790
9: -0.0036859, -0.0033367, -0.0036866, -0.0032855, -0.0004004, 0.0003499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.31 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002674, upper bound: 0.0002652
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002684, upper bound: 0.0002684
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009859, 0.0011011, 0.0009570, 0.0011005, -0.0001100, 0.0001408
1: 0.9936086, 0.9939758, 0.9936161, 0.9940484, -0.0003987, 0.0003023
2: -0.0066771, -0.0051973, -0.0065980, -0.0050705, -0.0013567, 0.0010559
3: 0.0038425, 0.0040217, 0.0038123, 0.0040165, -0.0001344, 0.0001812
4: 0.0025247, 0.0036942, 0.0024245, 0.0036317, -0.0009144, 0.0011097
5: 0.0061072, 0.0064729, 0.0060364, 0.0065003, -0.0003931, 0.0004365
6: -0.0014388, -0.0009251, -0.0014113, -0.0008811, -0.0004689, 0.0003619
7: -0.0082888, -0.0080090, -0.0083127, -0.0079587, -0.0003301, 0.0003038
8: 0.0050802, 0.0070245, 0.0049137, 0.0069206, -0.0011714, 0.0016876
9: -0.0036859, -0.0033367, -0.0036872, -0.0032442, -0.0004417, 0.0003505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.29 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002674, upper bound: 0.0002703
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002684, upper bound: 0.0002733
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010059, 0.0011008, 0.0009715, 0.0011012, -0.0000916, 0.0001253
1: 0.9936125, 0.9939255, 0.9936085, 0.9940118, -0.0003494, 0.0002703
2: -0.0066365, -0.0052848, -0.0066783, -0.0051342, -0.0011557, 0.0011097
3: 0.0038633, 0.0040191, 0.0038274, 0.0040218, -0.0001262, 0.0001570
4: 0.0025938, 0.0036621, 0.0024748, 0.0036952, -0.0009322, 0.0009590
5: 0.0061561, 0.0064540, 0.0060720, 0.0064866, -0.0003305, 0.0003821
6: -0.0014247, -0.0009555, -0.0014392, -0.0009032, -0.0003985, 0.0003819
7: -0.0082722, -0.0080166, -0.0083007, -0.0079843, -0.0002880, 0.0002841
8: 0.0051952, 0.0069711, 0.0049974, 0.0070260, -0.0013025, 0.0013972
9: -0.0036850, -0.0034005, -0.0036865, -0.0032907, -0.0003943, 0.0002860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.28 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002645, upper bound: 0.0002349
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002661, upper bound: 0.0002408
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010059, 0.0011008, 0.0009582, 0.0011006, -0.0000911, 0.0001386
1: 0.9936125, 0.9939255, 0.9936160, 0.9940456, -0.0003834, 0.0002642
2: -0.0066365, -0.0052848, -0.0065994, -0.0050758, -0.0012377, 0.0010621
3: 0.0038633, 0.0040191, 0.0038135, 0.0040166, -0.0001219, 0.0001715
4: 0.0025938, 0.0036621, 0.0024287, 0.0036328, -0.0009019, 0.0010237
5: 0.0061561, 0.0064540, 0.0060394, 0.0064992, -0.0003431, 0.0004147
6: -0.0014247, -0.0009555, -0.0014118, -0.0008830, -0.0004270, 0.0003650
7: -0.0082722, -0.0080166, -0.0083117, -0.0079608, -0.0003114, 0.0002951
8: 0.0051952, 0.0069711, 0.0049206, 0.0069224, -0.0012206, 0.0015049
9: -0.0036850, -0.0034005, -0.0036871, -0.0032480, -0.0004369, 0.0002866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.22 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002645, upper bound: 0.0002406
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002661, upper bound: 0.0002408
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009859, 0.0011011, 0.0009712, 0.0011012, -0.0001104, 0.0001266
1: 0.9936086, 0.9939758, 0.9936085, 0.9940126, -0.0003626, 0.0003069
2: -0.0066771, -0.0051973, -0.0066783, -0.0051329, -0.0012670, 0.0010892
3: 0.0038425, 0.0040217, 0.0038271, 0.0040218, -0.0001376, 0.0001660
4: 0.0025247, 0.0036942, 0.0024738, 0.0036952, -0.0009339, 0.0010388
5: 0.0061072, 0.0064729, 0.0060713, 0.0064869, -0.0003796, 0.0004017
6: -0.0014388, -0.0009251, -0.0014392, -0.0009028, -0.0004377, 0.0003737
7: -0.0082888, -0.0080090, -0.0083009, -0.0079838, -0.0003050, 0.0002920
8: 0.0050802, 0.0070245, 0.0049956, 0.0070261, -0.0012326, 0.0015697
9: -0.0036859, -0.0033367, -0.0036865, -0.0032897, -0.0003962, 0.0003499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.33 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002695, upper bound: 0.0002654
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002704, upper bound: 0.0002686
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009859, 0.0011011, 0.0009580, 0.0011006, -0.0001101, 0.0001398
1: 0.9936086, 0.9939758, 0.9936160, 0.9940461, -0.0003965, 0.0003033
2: -0.0066771, -0.0051973, -0.0065994, -0.0050747, -0.0013486, 0.0010665
3: 0.0038425, 0.0040217, 0.0038133, 0.0040166, -0.0001351, 0.0001804
4: 0.0025247, 0.0036942, 0.0024278, 0.0036328, -0.0009228, 0.0011033
5: 0.0061072, 0.0064729, 0.0060388, 0.0064994, -0.0003922, 0.0004342
6: -0.0014388, -0.0009251, -0.0014118, -0.0008826, -0.0004661, 0.0003654
7: -0.0082888, -0.0080090, -0.0083119, -0.0079604, -0.0003284, 0.0003030
8: 0.0050802, 0.0070245, 0.0049192, 0.0069224, -0.0011844, 0.0016770
9: -0.0036859, -0.0033367, -0.0036871, -0.0032473, -0.0004386, 0.0003505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.34 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002695, upper bound: 0.0002704
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002704, upper bound: 0.0002686
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009542, 0.0011005, 0.0009699, 0.0011011, -0.0001434, 0.0001275
1: 0.9936163, 0.9940557, 0.9936086, 0.9940161, -0.0003602, 0.0004023
2: -0.0065967, -0.0050580, -0.0066771, -0.0051271, -0.0012940, 0.0013922
3: 0.0038093, 0.0040164, 0.0038257, 0.0040217, -0.0001820, 0.0001633
4: 0.0024146, 0.0036307, 0.0024692, 0.0036942, -0.0011557, 0.0010832
5: 0.0060294, 0.0065030, 0.0060680, 0.0064881, -0.0004587, 0.0004350
6: -0.0014109, -0.0008768, -0.0014388, -0.0009008, -0.0004456, 0.0004799
7: -0.0083151, -0.0079536, -0.0083020, -0.0079814, -0.0003337, 0.0003484
8: 0.0048972, 0.0069189, 0.0049880, 0.0070245, -0.0016728, 0.0015270
9: -0.0036873, -0.0032350, -0.0036866, -0.0032855, -0.0004018, 0.0004515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002650, upper bound: 0.0002584
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003158, upper bound: 0.0003361
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009542, 0.0011005, 0.0009712, 0.0011012, -0.0001434, 0.0001261
1: 0.9936163, 0.9940557, 0.9936085, 0.9940126, -0.0003570, 0.0004026
2: -0.0065967, -0.0050580, -0.0066783, -0.0051329, -0.0012866, 0.0013946
3: 0.0038093, 0.0040164, 0.0038271, 0.0040218, -0.0001822, 0.0001621
4: 0.0024146, 0.0036307, 0.0024738, 0.0036952, -0.0011574, 0.0010773
5: 0.0060294, 0.0065030, 0.0060713, 0.0064869, -0.0004575, 0.0004318
6: -0.0014109, -0.0008768, -0.0014392, -0.0009028, -0.0004431, 0.0004807
7: -0.0083151, -0.0079536, -0.0083009, -0.0079838, -0.0003314, 0.0003473
8: 0.0048972, 0.0069189, 0.0049956, 0.0070261, -0.0016764, 0.0015173
9: -0.0036873, -0.0032350, -0.0036865, -0.0032897, -0.0003976, 0.0004515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002650, upper bound: 0.0002584
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003158, upper bound: 0.0003363
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009448, 0.0011001, 0.0009571, 0.0011005, -0.0001521, 0.0001398
1: 0.9936219, 0.9940792, 0.9936164, 0.9940485, -0.0003870, 0.0004187
2: -0.0065369, -0.0050170, -0.0065956, -0.0050709, -0.0012997, 0.0013706
3: 0.0037995, 0.0040125, 0.0038123, 0.0040163, -0.0001869, 0.0001732
4: 0.0023822, 0.0035834, 0.0024248, 0.0036298, -0.0011464, 0.0010971
5: 0.0060065, 0.0065119, 0.0060366, 0.0065003, -0.0004937, 0.0004753
6: -0.0013901, -0.0008625, -0.0014105, -0.0008812, -0.0004471, 0.0004720
7: -0.0083229, -0.0079371, -0.0083127, -0.0079588, -0.0003640, 0.0003755
8: 0.0048434, 0.0068403, 0.0049142, 0.0069175, -0.0016246, 0.0015196
9: -0.0036877, -0.0032051, -0.0036872, -0.0032445, -0.0004432, 0.0004820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.50 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002676, upper bound: 0.0002213
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002829, upper bound: 0.0002700
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009448, 0.0011001, 0.0009581, 0.0011005, -0.0001522, 0.0001388
1: 0.9936219, 0.9940792, 0.9936162, 0.9940459, -0.0003844, 0.0004189
2: -0.0065369, -0.0050170, -0.0065970, -0.0050751, -0.0012920, 0.0013735
3: 0.0037995, 0.0040125, 0.0038134, 0.0040164, -0.0001870, 0.0001721
4: 0.0023822, 0.0035834, 0.0024281, 0.0036309, -0.0011482, 0.0010910
5: 0.0060065, 0.0065119, 0.0060390, 0.0064993, -0.0004928, 0.0004729
6: -0.0013901, -0.0008625, -0.0014110, -0.0008827, -0.0004444, 0.0004730
7: -0.0083229, -0.0079371, -0.0083119, -0.0079605, -0.0003624, 0.0003748
8: 0.0048434, 0.0068403, 0.0049197, 0.0069193, -0.0016274, 0.0015095
9: -0.0036877, -0.0032051, -0.0036871, -0.0032475, -0.0004402, 0.0004820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.53 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002676, upper bound: 0.0002217
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002829, upper bound: 0.0002702
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009542, 0.0011005, 0.0009570, 0.0011005, -0.0001428, 0.0001402
1: 0.9936163, 0.9940557, 0.9936161, 0.9940484, -0.0003912, 0.0003961
2: -0.0065967, -0.0050580, -0.0065980, -0.0050705, -0.0013457, 0.0013270
3: 0.0038093, 0.0040164, 0.0038123, 0.0040165, -0.0001776, 0.0001759
4: 0.0024146, 0.0036307, 0.0024245, 0.0036317, -0.0011119, 0.0011276
5: 0.0060294, 0.0065030, 0.0060364, 0.0065003, -0.0004709, 0.0004666
6: -0.0014109, -0.0008768, -0.0014113, -0.0008811, -0.0004634, 0.0004568
7: -0.0083151, -0.0079536, -0.0083127, -0.0079587, -0.0003565, 0.0003591
8: 0.0048972, 0.0069189, 0.0049137, 0.0069206, -0.0015669, 0.0015906
9: -0.0036873, -0.0032350, -0.0036872, -0.0032442, -0.0004431, 0.0004521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003206, upper bound: 0.0003206
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003206, upper bound: 0.0003206
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009542, 0.0011005, 0.0009580, 0.0011006, -0.0001429, 0.0001392
1: 0.9936163, 0.9940557, 0.9936160, 0.9940461, -0.0003887, 0.0003963
2: -0.0065967, -0.0050580, -0.0065994, -0.0050747, -0.0013379, 0.0013298
3: 0.0038093, 0.0040164, 0.0038133, 0.0040166, -0.0001778, 0.0001749
4: 0.0024146, 0.0036307, 0.0024278, 0.0036328, -0.0011138, 0.0011214
5: 0.0060294, 0.0065030, 0.0060388, 0.0064994, -0.0004700, 0.0004643
6: -0.0014109, -0.0008768, -0.0014118, -0.0008826, -0.0004607, 0.0004578
7: -0.0083151, -0.0079536, -0.0083119, -0.0079604, -0.0003548, 0.0003583
8: 0.0048972, 0.0069189, 0.0049192, 0.0069224, -0.0015697, 0.0015803
9: -0.0036873, -0.0032350, -0.0036871, -0.0032473, -0.0004400, 0.0004521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003206, upper bound: 0.0003289
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003206, upper bound: 0.0003653
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009696, 0.0011009, 0.0010038, 0.0011008, -0.0001269, 0.0000935
1: 0.9936112, 0.9940168, 0.9936131, 0.9939306, -0.0002742, 0.0003499
2: -0.0066492, -0.0051255, -0.0066298, -0.0052758, -0.0011065, 0.0011141
3: 0.0038254, 0.0040199, 0.0038612, 0.0040186, -0.0001560, 0.0001275
4: 0.0024680, 0.0036722, 0.0025868, 0.0036568, -0.0009268, 0.0009286
5: 0.0060671, 0.0064885, 0.0061511, 0.0064560, -0.0003888, 0.0003373
6: -0.0014291, -0.0009002, -0.0014224, -0.0009524, -0.0003809, 0.0003841
7: -0.0083023, -0.0079808, -0.0082739, -0.0080179, -0.0002844, 0.0002931
8: 0.0049859, 0.0069878, 0.0051834, 0.0069623, -0.0013498, 0.0013015
9: -0.0036866, -0.0032843, -0.0036851, -0.0033940, -0.0002926, 0.0004007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.57 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002729
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002943
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009680, 0.0011009, 0.0009834, 0.0011009, -0.0001294, 0.0001130
1: 0.9936112, 0.9940209, 0.9936113, 0.9939823, -0.0003143, 0.0003652
2: -0.0066492, -0.0051186, -0.0066482, -0.0051860, -0.0011263, 0.0012255
3: 0.0038237, 0.0040199, 0.0038398, 0.0040198, -0.0001655, 0.0001411
4: 0.0024625, 0.0036722, 0.0025158, 0.0036714, -0.0010093, 0.0009607
5: 0.0060633, 0.0064899, 0.0061009, 0.0064754, -0.0004121, 0.0003890
6: -0.0014291, -0.0008978, -0.0014287, -0.0009212, -0.0003868, 0.0004231
7: -0.0083036, -0.0079780, -0.0082909, -0.0080051, -0.0002985, 0.0003129
8: 0.0049769, 0.0069878, 0.0050654, 0.0069865, -0.0015172, 0.0012858
9: -0.0036867, -0.0032793, -0.0036860, -0.0033285, -0.0003582, 0.0004067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.53 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002931, upper bound: 0.0002729
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002931, upper bound: 0.0002974
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009594, 0.0011006, 0.0010038, 0.0011008, -0.0001371, 0.0000932
1: 0.9936160, 0.9940427, 0.9936131, 0.9939306, -0.0002701, 0.0003762
2: -0.0065994, -0.0050807, -0.0066298, -0.0052758, -0.0010821, 0.0011787
3: 0.0038147, 0.0040166, 0.0038612, 0.0040186, -0.0001675, 0.0001247
4: 0.0024326, 0.0036328, 0.0025868, 0.0036568, -0.0009779, 0.0009173
5: 0.0060421, 0.0064981, 0.0061511, 0.0064560, -0.0004138, 0.0003470
6: -0.0014118, -0.0008847, -0.0014224, -0.0009524, -0.0003720, 0.0004065
7: -0.0083108, -0.0079628, -0.0082739, -0.0080179, -0.0002929, 0.0003112
8: 0.0049271, 0.0069224, 0.0051834, 0.0069623, -0.0014347, 0.0012481
9: -0.0036871, -0.0032516, -0.0036851, -0.0033940, -0.0002931, 0.0004334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002273, upper bound: 0.0002266
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002790, upper bound: 0.0003074
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009580, 0.0011006, 0.0009834, 0.0011009, -0.0001393, 0.0001127
1: 0.9936160, 0.9940461, 0.9936113, 0.9939823, -0.0003106, 0.0003911
2: -0.0065994, -0.0050747, -0.0066482, -0.0051860, -0.0011067, 0.0012888
3: 0.0038133, 0.0040166, 0.0038398, 0.0040198, -0.0001769, 0.0001385
4: 0.0024278, 0.0036328, 0.0025158, 0.0036714, -0.0010593, 0.0009545
5: 0.0060388, 0.0064994, 0.0061009, 0.0064754, -0.0004366, 0.0003985
6: -0.0014118, -0.0008826, -0.0014287, -0.0009212, -0.0003794, 0.0004450
7: -0.0083119, -0.0079604, -0.0082909, -0.0080051, -0.0003068, 0.0003305
8: 0.0049192, 0.0069224, 0.0050654, 0.0069865, -0.0016003, 0.0012371
9: -0.0036871, -0.0032473, -0.0036860, -0.0033285, -0.0003586, 0.0004387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002554, upper bound: 0.0002358
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003033, upper bound: 0.0003126
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009697, 0.0011009, 0.0010044, 0.0011008, -0.0001268, 0.0000930
1: 0.9936112, 0.9940164, 0.9936129, 0.9939291, -0.0002731, 0.0003500
2: -0.0066492, -0.0051262, -0.0066324, -0.0052783, -0.0011000, 0.0011163
3: 0.0038255, 0.0040199, 0.0038618, 0.0040188, -0.0001561, 0.0001271
4: 0.0024685, 0.0036722, 0.0025887, 0.0036589, -0.0009286, 0.0009234
5: 0.0060675, 0.0064883, 0.0061525, 0.0064554, -0.0003879, 0.0003358
6: -0.0014291, -0.0009004, -0.0014233, -0.0009533, -0.0003786, 0.0003849
7: -0.0083022, -0.0079811, -0.0082734, -0.0080174, -0.0002848, 0.0002924
8: 0.0049868, 0.0069878, 0.0051867, 0.0069659, -0.0013532, 0.0012997
9: -0.0036866, -0.0032848, -0.0036850, -0.0033958, -0.0002908, 0.0004002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.56 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002739
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002975
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009680, 0.0011009, 0.0009850, 0.0011009, -0.0001294, 0.0001114
1: 0.9936112, 0.9940209, 0.9936112, 0.9939779, -0.0003104, 0.0003653
2: -0.0066492, -0.0051186, -0.0066492, -0.0051931, -0.0011148, 0.0012285
3: 0.0038237, 0.0040199, 0.0038415, 0.0040199, -0.0001656, 0.0001395
4: 0.0024625, 0.0036722, 0.0025214, 0.0036722, -0.0010111, 0.0009501
5: 0.0060633, 0.0064899, 0.0061049, 0.0064738, -0.0004106, 0.0003850
6: -0.0014291, -0.0008978, -0.0014291, -0.0009237, -0.0003828, 0.0004241
7: -0.0083036, -0.0079780, -0.0082896, -0.0080080, -0.0002957, 0.0003116
8: 0.0049769, 0.0069878, 0.0050747, 0.0069878, -0.0015183, 0.0012733
9: -0.0036867, -0.0032793, -0.0036859, -0.0033336, -0.0003530, 0.0004066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 5.55 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002978, upper bound: 0.0002739
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002978, upper bound: 0.0003008
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009595, 0.0011006, 0.0010044, 0.0011008, -0.0001370, 0.0000926
1: 0.9936160, 0.9940423, 0.9936129, 0.9939291, -0.0002687, 0.0003763
2: -0.0065994, -0.0050812, -0.0066324, -0.0052783, -0.0010758, 0.0011780
3: 0.0038148, 0.0040166, 0.0038618, 0.0040188, -0.0001676, 0.0001241
4: 0.0024329, 0.0036328, 0.0025887, 0.0036589, -0.0009773, 0.0009120
5: 0.0060424, 0.0064980, 0.0061525, 0.0064554, -0.0004131, 0.0003455
6: -0.0014118, -0.0008848, -0.0014233, -0.0009533, -0.0003698, 0.0004063
7: -0.0083107, -0.0079629, -0.0082734, -0.0080174, -0.0002933, 0.0003105
8: 0.0049277, 0.0069224, 0.0051867, 0.0069659, -0.0014342, 0.0012410
9: -0.0036871, -0.0032520, -0.0036850, -0.0033958, -0.0002912, 0.0004331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002273, upper bound: 0.0002266
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002800, upper bound: 0.0003091
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009580, 0.0011006, 0.0009850, 0.0011009, -0.0001394, 0.0001111
1: 0.9936160, 0.9940461, 0.9936112, 0.9939779, -0.0003064, 0.0003910
2: -0.0065994, -0.0050747, -0.0066492, -0.0051931, -0.0010922, 0.0012886
3: 0.0038133, 0.0040166, 0.0038415, 0.0040199, -0.0001768, 0.0001367
4: 0.0024278, 0.0036328, 0.0025214, 0.0036722, -0.0010585, 0.0009427
5: 0.0060388, 0.0064994, 0.0061049, 0.0064738, -0.0004351, 0.0003945
6: -0.0014118, -0.0008826, -0.0014291, -0.0009237, -0.0003745, 0.0004450
7: -0.0083119, -0.0079604, -0.0082896, -0.0080080, -0.0003040, 0.0003292
8: 0.0049192, 0.0069224, 0.0050747, 0.0069878, -0.0015971, 0.0012213
9: -0.0036871, -0.0032473, -0.0036859, -0.0033336, -0.0003535, 0.0004386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002565, upper bound: 0.0002368
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003081, upper bound: 0.0003151
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010068, 0.0011009, 0.0009701, 0.0011011, -0.0000906, 0.0001268
1: 0.9936122, 0.9939228, 0.9936086, 0.9940154, -0.0003534, 0.0002678
2: -0.0066393, -0.0052890, -0.0066771, -0.0051280, -0.0011655, 0.0011119
3: 0.0038644, 0.0040192, 0.0038260, 0.0040217, -0.0001253, 0.0001590
4: 0.0025972, 0.0036643, 0.0024699, 0.0036942, -0.0009342, 0.0009662
5: 0.0061585, 0.0064531, 0.0060685, 0.0064879, -0.0003294, 0.0003846
6: -0.0014256, -0.0009570, -0.0014388, -0.0009011, -0.0004019, 0.0003827
7: -0.0082714, -0.0080161, -0.0083019, -0.0079818, -0.0002896, 0.0002857
8: 0.0052008, 0.0069748, 0.0049892, 0.0070245, -0.0013048, 0.0014142
9: -0.0036849, -0.0034036, -0.0036866, -0.0032861, -0.0003988, 0.0002830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.31 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002632, upper bound: 0.0002349
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002649, upper bound: 0.0002409
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010068, 0.0011009, 0.0009572, 0.0011005, -0.0000901, 0.0001397
1: 0.9936122, 0.9939228, 0.9936161, 0.9940481, -0.0003864, 0.0002620
2: -0.0066393, -0.0052890, -0.0065980, -0.0050712, -0.0012485, 0.0010617
3: 0.0038644, 0.0040192, 0.0038124, 0.0040165, -0.0001212, 0.0001727
4: 0.0025972, 0.0036643, 0.0024250, 0.0036317, -0.0009021, 0.0010318
5: 0.0061585, 0.0064531, 0.0060368, 0.0065002, -0.0003417, 0.0004163
6: -0.0014256, -0.0009570, -0.0014113, -0.0008814, -0.0004307, 0.0003649
7: -0.0082714, -0.0080161, -0.0083126, -0.0079589, -0.0003125, 0.0002965
8: 0.0052008, 0.0069748, 0.0049146, 0.0069206, -0.0012248, 0.0015233
9: -0.0036849, -0.0034036, -0.0036872, -0.0032447, -0.0004402, 0.0002835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.25 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002632, upper bound: 0.0002406
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002649, upper bound: 0.0002462
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009875, 0.0011012, 0.0009699, 0.0011011, -0.0001087, 0.0001279
1: 0.9936085, 0.9939718, 0.9936086, 0.9940161, -0.0003657, 0.0003018
2: -0.0066783, -0.0052042, -0.0066771, -0.0051271, -0.0012718, 0.0010869
3: 0.0038441, 0.0040218, 0.0038257, 0.0040217, -0.0001352, 0.0001674
4: 0.0025301, 0.0036952, 0.0024692, 0.0036942, -0.0009335, 0.0010440
5: 0.0061111, 0.0064715, 0.0060680, 0.0064881, -0.0003770, 0.0004035
6: -0.0014392, -0.0009275, -0.0014388, -0.0009008, -0.0004392, 0.0003729
7: -0.0082875, -0.0080088, -0.0083020, -0.0079814, -0.0003061, 0.0002933
8: 0.0050893, 0.0070261, 0.0049880, 0.0070245, -0.0012275, 0.0015764
9: -0.0036858, -0.0033417, -0.0036866, -0.0032855, -0.0004003, 0.0003449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.23 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002679, upper bound: 0.0002666
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002686, upper bound: 0.0002704
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009875, 0.0011012, 0.0009570, 0.0011005, -0.0001085, 0.0001408
1: 0.9936085, 0.9939718, 0.9936161, 0.9940484, -0.0003986, 0.0002985
2: -0.0066783, -0.0052042, -0.0065980, -0.0050705, -0.0013545, 0.0010661
3: 0.0038441, 0.0040218, 0.0038123, 0.0040165, -0.0001329, 0.0001811
4: 0.0025301, 0.0036952, 0.0024245, 0.0036317, -0.0009224, 0.0011094
5: 0.0061111, 0.0064715, 0.0060364, 0.0065003, -0.0003893, 0.0004350
6: -0.0014392, -0.0009275, -0.0014113, -0.0008811, -0.0004680, 0.0003654
7: -0.0082875, -0.0080088, -0.0083127, -0.0079587, -0.0003288, 0.0003040
8: 0.0050893, 0.0070261, 0.0049137, 0.0069206, -0.0011848, 0.0016851
9: -0.0036858, -0.0033417, -0.0036872, -0.0032442, -0.0004416, 0.0003455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.43 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002679, upper bound: 0.0002717
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002686, upper bound: 0.0002704
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0010068, 0.0011009, 0.0009717, 0.0011012, -0.0000906, 0.0001252
1: 0.9936122, 0.9939228, 0.9936085, 0.9940116, -0.0003495, 0.0002679
2: -0.0066393, -0.0052890, -0.0066783, -0.0051347, -0.0011578, 0.0011071
3: 0.0038644, 0.0040192, 0.0038276, 0.0040218, -0.0001252, 0.0001571
4: 0.0025972, 0.0036643, 0.0024752, 0.0036952, -0.0009297, 0.0009600
5: 0.0061585, 0.0064531, 0.0060723, 0.0064865, -0.0003280, 0.0003809
6: -0.0014256, -0.0009570, -0.0014392, -0.0009034, -0.0003992, 0.0003810
7: -0.0082714, -0.0080161, -0.0083006, -0.0079845, -0.0002870, 0.0002845
8: 0.0052008, 0.0069748, 0.0049980, 0.0070261, -0.0012986, 0.0014010
9: -0.0036849, -0.0034036, -0.0036865, -0.0032910, -0.0003939, 0.0002829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.29 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002654, upper bound: 0.0002349
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002668, upper bound: 0.0002415
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0010068, 0.0011009, 0.0009583, 0.0011006, -0.0000901, 0.0001385
1: 0.9936122, 0.9939228, 0.9936160, 0.9940453, -0.0003835, 0.0002619
2: -0.0066393, -0.0052890, -0.0065994, -0.0050762, -0.0012410, 0.0010584
3: 0.0038644, 0.0040192, 0.0038136, 0.0040166, -0.0001211, 0.0001715
4: 0.0025972, 0.0036643, 0.0024290, 0.0036328, -0.0008988, 0.0010258
5: 0.0061585, 0.0064531, 0.0060396, 0.0064991, -0.0003406, 0.0004135
6: -0.0014256, -0.0009570, -0.0014118, -0.0008831, -0.0004281, 0.0003637
7: -0.0082714, -0.0080161, -0.0083117, -0.0079609, -0.0003105, 0.0002955
8: 0.0052008, 0.0069748, 0.0049211, 0.0069224, -0.0012169, 0.0015104
9: -0.0036849, -0.0034036, -0.0036871, -0.0032483, -0.0004366, 0.0002835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.38 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002654, upper bound: 0.0002406
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002668, upper bound: 0.0002466
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0009875, 0.0011012, 0.0009712, 0.0011012, -0.0001088, 0.0001266
1: 0.9936085, 0.9939718, 0.9936085, 0.9940126, -0.0003627, 0.0003022
2: -0.0066783, -0.0052042, -0.0066783, -0.0051329, -0.0012675, 0.0010755
3: 0.0038441, 0.0040218, 0.0038271, 0.0040218, -0.0001355, 0.0001661
4: 0.0025301, 0.0036952, 0.0024738, 0.0036952, -0.0009239, 0.0010389
5: 0.0061111, 0.0064715, 0.0060713, 0.0064869, -0.0003758, 0.0004002
6: -0.0014392, -0.0009275, -0.0014392, -0.0009028, -0.0004379, 0.0003689
7: -0.0082875, -0.0080088, -0.0083009, -0.0079838, -0.0003037, 0.0002922
8: 0.0050893, 0.0070261, 0.0049956, 0.0070261, -0.0012159, 0.0015708
9: -0.0036858, -0.0033417, -0.0036865, -0.0032897, -0.0003961, 0.0003448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.27 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002703, upper bound: 0.0002676
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002709, upper bound: 0.0002709
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009875, 0.0011012, 0.0009580, 0.0011006, -0.0001085, 0.0001398
1: 0.9936085, 0.9939718, 0.9936160, 0.9940461, -0.0003965, 0.0002988
2: -0.0066783, -0.0052042, -0.0065994, -0.0050747, -0.0013503, 0.0010547
3: 0.0038441, 0.0040218, 0.0038133, 0.0040166, -0.0001332, 0.0001804
4: 0.0025301, 0.0036952, 0.0024278, 0.0036328, -0.0009131, 0.0011043
5: 0.0061111, 0.0064715, 0.0060388, 0.0064994, -0.0003884, 0.0004327
6: -0.0014392, -0.0009275, -0.0014118, -0.0008826, -0.0004666, 0.0003615
7: -0.0082875, -0.0080088, -0.0083119, -0.0079604, -0.0003271, 0.0003032
8: 0.0050893, 0.0070261, 0.0049192, 0.0069224, -0.0011720, 0.0016795
9: -0.0036858, -0.0033417, -0.0036871, -0.0032473, -0.0004385, 0.0003454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 7.29 seconds

### Candidate
type: A, layer: 3, pos: 80

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002703, upper bound: 0.0002723
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002709, upper bound: 0.0002759
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0009687, 0.0011009, 0.0009925, 0.0011004, -0.0001274, 0.0001048
1: 0.9936112, 0.9940191, 0.9936186, 0.9939590, -0.0003023, 0.0003479
2: -0.0066492, -0.0051218, -0.0065718, -0.0052263, -0.0011738, 0.0010971
3: 0.0038245, 0.0040199, 0.0038494, 0.0040148, -0.0001539, 0.0001392
4: 0.0024650, 0.0036722, 0.0025476, 0.0036110, -0.0009228, 0.0009822
5: 0.0060651, 0.0064893, 0.0061234, 0.0064667, -0.0004016, 0.0003658
6: -0.0014291, -0.0008989, -0.0014022, -0.0009352, -0.0004042, 0.0003776
7: -0.0083030, -0.0079793, -0.0082833, -0.0080213, -0.0002817, 0.0003040
8: 0.0049811, 0.0069878, 0.0051184, 0.0068862, -0.0013004, 0.0013891
9: -0.0036866, -0.0032816, -0.0036856, -0.0033579, -0.0003288, 0.0004040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002175, upper bound: 0.0002435
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 6.09 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002936, upper bound: 0.0002801
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0002936, upper bound: 0.0003053
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0009680, 0.0011009, 0.0009733, 0.0011005, -0.0001290, 0.0001231
1: 0.9936112, 0.9940209, 0.9936161, 0.9940075, -0.0003395, 0.0003610
2: -0.0066492, -0.0051186, -0.0065980, -0.0051420, -0.0011795, 0.0011998
3: 0.0038237, 0.0040199, 0.0038293, 0.0040165, -0.0001626, 0.0001515
4: 0.0024625, 0.0036722, 0.0024810, 0.0036317, -0.0009938, 0.0010029
5: 0.0060633, 0.0064899, 0.0060763, 0.0064849, -0.0004216, 0.0004136
6: -0.0014291, -0.0008978, -0.0014113, -0.0009059, -0.0004053, 0.0004138
7: -0.0083036, -0.0079780, -0.0082992, -0.0079874, -0.0003163, 0.0003212
8: 0.0049769, 0.0069878, 0.0050076, 0.0069206, -0.0014604, 0.0013558
9: -0.0036867, -0.0032793, -0.0036864, -0.0032963, -0.0003903, 0.0004071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002204, upper bound: 0.0002435
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 136
type: A, layer: 3, pos: 81

Time for candidate selection: 6.11 seconds

### Candidate
type: A, layer: 3, pos: 73

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003082, upper bound: 0.0002801
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0003082, upper bound: 0.0002801
time: 0.57 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 8.45 seconds
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002725, upper bound: 0.0002725
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002725, upper bound: 0.0002924
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002924, upper bound: 0.0002725
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002924, upper bound: 0.0002951
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002281, upper bound: 0.0002266
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002787, upper bound: 0.0003070
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002564, upper bound: 0.0002358
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003032, upper bound: 0.0003117
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002729, upper bound: 0.0002734
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002729, upper bound: 0.0002931
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002943, upper bound: 0.0002734
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002943, upper bound: 0.0002958
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002281, upper bound: 0.0002266
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002791, upper bound: 0.0003078
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002589, upper bound: 0.0002373
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003062, upper bound: 0.0003129
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002632, upper bound: 0.0002349
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002650, upper bound: 0.0002408
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002632, upper bound: 0.0002349
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002650, upper bound: 0.0002458
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002674, upper bound: 0.0002652
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002684, upper bound: 0.0002684
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002674, upper bound: 0.0002703
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002684, upper bound: 0.0002733
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002645, upper bound: 0.0002349
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002661, upper bound: 0.0002408
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002645, upper bound: 0.0002406
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002661, upper bound: 0.0002408
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002695, upper bound: 0.0002654
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002704, upper bound: 0.0002686
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002695, upper bound: 0.0002704
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002704, upper bound: 0.0002686
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002650, upper bound: 0.0002584
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003158, upper bound: 0.0003361
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002650, upper bound: 0.0002584
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003158, upper bound: 0.0003363
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002676, upper bound: 0.0002213
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002829, upper bound: 0.0002700
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002676, upper bound: 0.0002217
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002829, upper bound: 0.0002702
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003206, upper bound: 0.0003206
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003206, upper bound: 0.0003206
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003206, upper bound: 0.0003289
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003206, upper bound: 0.0003653
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002729
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002943
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002931, upper bound: 0.0002729
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002931, upper bound: 0.0002974
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002273, upper bound: 0.0002266
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002790, upper bound: 0.0003074
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002554, upper bound: 0.0002358
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003033, upper bound: 0.0003126
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002739
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002975
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002978, upper bound: 0.0002739
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002978, upper bound: 0.0003008
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002273, upper bound: 0.0002266
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002800, upper bound: 0.0003091
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002565, upper bound: 0.0002368
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003081, upper bound: 0.0003151
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002632, upper bound: 0.0002349
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002649, upper bound: 0.0002409
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002632, upper bound: 0.0002406
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002649, upper bound: 0.0002462
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002679, upper bound: 0.0002666
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002686, upper bound: 0.0002704
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002679, upper bound: 0.0002717
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002686, upper bound: 0.0002704
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002654, upper bound: 0.0002349
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002668, upper bound: 0.0002415
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002654, upper bound: 0.0002406
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002668, upper bound: 0.0002466
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002703, upper bound: 0.0002676
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002709, upper bound: 0.0002709
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002703, upper bound: 0.0002723
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002709, upper bound: 0.0002759
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002936, upper bound: 0.0002801
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0002936, upper bound: 0.0003053
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003082, upper bound: 0.0002801
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.45
Output dim: 1, lower bound: -0.0003082, upper bound: 0.0002801
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.45
Output dim: 1, lower bound: -0.0002734, upper bound: 0.0002943
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.45
Output dim: 1, lower bound: -0.0002958, upper bound: 0.0003396
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.45
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0003073
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.45
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0003084
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.45
Output dim: 1, lower bound: -0.0002743, upper bound: 0.0002975
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.45
Output dim: 1, lower bound: -0.0003011, upper bound: 0.0002942

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.70 + 597.45 = 600.15 seconds
