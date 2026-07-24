## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00162


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036630, 0.0036630)
1: (-0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009127, 0.0009127)
2: (-0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0048369, 0.0048369)
3: (0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0022016, 0.0022016)
4: (-0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009362, 0.0009362)
5: (-0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0060836, 0.0060836)
6: (0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015441, 0.0015441)
7: (0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039950, 0.0039950)
8: (0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0021009, 0.0021009)
9: (-0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0024361, 0.0024361)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 2.02 = 3.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0020250, upper bound: 0.0020250

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019058, upper bound: 0.0018018
time: 1.20 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019058, upper bound: 0.0019058
time: 1.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.42
Output dim: 0, lower bound: -0.0019058, upper bound: 0.0018018
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.42
Output dim: 0, lower bound: -0.0019058, upper bound: 0.0019058

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 1.0013189, 1.0055922, 1.0013155, 1.0058299, -0.0035741, 0.0033343
1: -0.0009353, 0.0001295, -0.0009362, 0.0001887, -0.0008906, 0.0008308
2: -0.0107402, -0.0050974, -0.0110541, -0.0050928, -0.0044030, 0.0047196
3: 0.0010470, 0.0036153, 0.0010449, 0.0037582, -0.0021482, 0.0020040
4: -0.0015508, -0.0004587, -0.0016116, -0.0004578, -0.0008522, 0.0009135
5: -0.0145489, -0.0074517, -0.0149437, -0.0074459, -0.0055378, 0.0059360
6: 0.0034321, 0.0052335, 0.0034307, 0.0053337, -0.0015066, 0.0014055
7: 0.0057424, 0.0104030, 0.0057386, 0.0106623, -0.0038981, 0.0036366
8: 0.0034557, 0.0059067, 0.0034537, 0.0060430, -0.0020500, 0.0019124
9: -0.0087129, -0.0058709, -0.0088710, -0.0058686, -0.0022176, 0.0023770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018018, upper bound: 0.0018018
time: 1.16 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018018, upper bound: 0.0018018
time: 1.14 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 1.0010102, 1.0056772, 1.0013163, 1.0058266, -0.0039962, 0.0034210
1: -0.0010122, 0.0001507, -0.0009360, 0.0001879, -0.0009958, 0.0008524
2: -0.0108524, -0.0046897, -0.0110496, -0.0050940, -0.0045174, 0.0052770
3: 0.0008614, 0.0036664, 0.0010454, 0.0037562, -0.0024018, 0.0020561
4: -0.0015726, -0.0003798, -0.0016107, -0.0004580, -0.0008743, 0.0010213
5: -0.0146900, -0.0069389, -0.0149381, -0.0074474, -0.0056817, 0.0066370
6: 0.0033020, 0.0052693, 0.0034311, 0.0053323, -0.0016845, 0.0014421
7: 0.0054057, 0.0104957, 0.0057396, 0.0106586, -0.0043584, 0.0037311
8: 0.0032786, 0.0059554, 0.0034542, 0.0060411, -0.0022921, 0.0019621
9: -0.0087695, -0.0056656, -0.0088688, -0.0058692, -0.0022752, 0.0026578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018017, upper bound: 0.0019058
time: 1.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018017, upper bound: 0.0019058
time: 1.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.97
Output dim: 0, lower bound: -0.0018018, upper bound: 0.0018018
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.97
Output dim: 0, lower bound: -0.0018018, upper bound: 0.0018018
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.97
Output dim: 0, lower bound: -0.0018017, upper bound: 0.0019058
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.97
Output dim: 0, lower bound: -0.0018017, upper bound: 0.0019058

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 1.0013189, 1.0055922, 1.0013189, 1.0055922, -0.0033309, 0.0033309
1: -0.0009353, 0.0001295, -0.0009353, 0.0001295, -0.0008300, 0.0008300
2: -0.0107402, -0.0050974, -0.0107402, -0.0050974, -0.0043984, 0.0043984
3: 0.0010470, 0.0036153, 0.0010470, 0.0036153, -0.0020020, 0.0020020
4: -0.0015508, -0.0004587, -0.0015508, -0.0004587, -0.0008513, 0.0008513
5: -0.0145489, -0.0074517, -0.0145489, -0.0074517, -0.0055321, 0.0055321
6: 0.0034321, 0.0052335, 0.0034321, 0.0052335, -0.0014041, 0.0014041
7: 0.0057424, 0.0104030, 0.0057424, 0.0104030, -0.0036328, 0.0036328
8: 0.0034557, 0.0059067, 0.0034557, 0.0059067, -0.0019105, 0.0019105
9: -0.0087129, -0.0058709, -0.0087129, -0.0058709, -0.0022153, 0.0022153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017238, upper bound: 0.0017284
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017427, upper bound: 0.0017290
time: 1.17 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013189, 1.0055922, 1.0010102, 1.0056772, -0.0034724, 0.0037400
1: -0.0009353, 0.0001295, -0.0010122, 0.0001507, -0.0008652, 0.0009319
2: -0.0107402, -0.0050974, -0.0108524, -0.0046897, -0.0049386, 0.0045853
3: 0.0010470, 0.0036153, 0.0008614, 0.0036664, -0.0020870, 0.0022478
4: -0.0015508, -0.0004587, -0.0015726, -0.0003798, -0.0009559, 0.0008875
5: -0.0145489, -0.0074517, -0.0146900, -0.0069389, -0.0062114, 0.0057670
6: 0.0034321, 0.0052335, 0.0033020, 0.0052693, -0.0014637, 0.0015765
7: 0.0057424, 0.0104030, 0.0054057, 0.0104957, -0.0037871, 0.0040790
8: 0.0034557, 0.0059067, 0.0032786, 0.0059554, -0.0019916, 0.0021451
9: -0.0087129, -0.0058709, -0.0087695, -0.0056656, -0.0024873, 0.0023094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017238, upper bound: 0.0017284
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017427, upper bound: 0.0017290
time: 1.18 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 1.0010102, 1.0056772, 1.0013189, 1.0055922, -0.0037400, 0.0034724
1: -0.0010122, 0.0001507, -0.0009353, 0.0001295, -0.0009319, 0.0008652
2: -0.0108524, -0.0046897, -0.0107402, -0.0050974, -0.0045853, 0.0049386
3: 0.0008614, 0.0036664, 0.0010470, 0.0036153, -0.0022478, 0.0020870
4: -0.0015726, -0.0003798, -0.0015508, -0.0004587, -0.0008875, 0.0009559
5: -0.0146900, -0.0069389, -0.0145489, -0.0074517, -0.0057670, 0.0062114
6: 0.0033020, 0.0052693, 0.0034321, 0.0052335, -0.0015765, 0.0014637
7: 0.0054057, 0.0104957, 0.0057424, 0.0104030, -0.0040790, 0.0037871
8: 0.0032786, 0.0059554, 0.0034557, 0.0059067, -0.0021451, 0.0019916
9: -0.0087695, -0.0056656, -0.0087129, -0.0058709, -0.0023094, 0.0024873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017127, upper bound: 0.0018331
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017290, upper bound: 0.0018348
time: 1.21 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 1.0010102, 1.0056772, 1.0010102, 1.0056772, -0.0034483, 0.0034483
1: -0.0010122, 0.0001507, -0.0010122, 0.0001507, -0.0008592, 0.0008592
2: -0.0108524, -0.0046897, -0.0108524, -0.0046897, -0.0045534, 0.0045534
3: 0.0008614, 0.0036664, 0.0008614, 0.0036664, -0.0020725, 0.0020725
4: -0.0015726, -0.0003798, -0.0015726, -0.0003798, -0.0008813, 0.0008813
5: -0.0146900, -0.0069389, -0.0146900, -0.0069389, -0.0057270, 0.0057270
6: 0.0033020, 0.0052693, 0.0033020, 0.0052693, -0.0014536, 0.0014536
7: 0.0054057, 0.0104957, 0.0054057, 0.0104957, -0.0037608, 0.0037608
8: 0.0032786, 0.0059554, 0.0032786, 0.0059554, -0.0019778, 0.0019778
9: -0.0087695, -0.0056656, -0.0087695, -0.0056656, -0.0022933, 0.0022933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017127, upper bound: 0.0018332
time: 1.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017290, upper bound: 0.0018348
time: 1.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.82 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0017238, upper bound: 0.0017284
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0017427, upper bound: 0.0017290
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0017238, upper bound: 0.0017284
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0017427, upper bound: 0.0017290
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0017127, upper bound: 0.0018331
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0017290, upper bound: 0.0018348
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0017127, upper bound: 0.0018332
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0017290, upper bound: 0.0018348

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0014017, 1.0055848, -0.0030104, 0.0032355
1: -0.0008787, 0.0001534, -0.0009147, 0.0001276, -0.0007501, 0.0008062
2: -0.0108669, -0.0053976, -0.0107304, -0.0052065, -0.0042725, 0.0039752
3: 0.0011836, 0.0036730, 0.0010967, 0.0036109, -0.0018093, 0.0019446
4: -0.0015754, -0.0005168, -0.0015490, -0.0004798, -0.0008269, 0.0007694
5: -0.0147082, -0.0078293, -0.0145366, -0.0075890, -0.0053736, 0.0049998
6: 0.0035280, 0.0052739, 0.0034670, 0.0052304, -0.0012690, 0.0013639
7: 0.0059904, 0.0105077, 0.0058325, 0.0103949, -0.0032833, 0.0035288
8: 0.0035861, 0.0059617, 0.0035031, 0.0059025, -0.0017266, 0.0018558
9: -0.0087768, -0.0060222, -0.0087080, -0.0059259, -0.0021518, 0.0020021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017313
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017496
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0014151, 1.0055836, 1.0013340, 1.0055909, -0.0030929, 0.0032854
1: -0.0009113, 0.0001273, -0.0009316, 0.0001291, -0.0007707, 0.0008186
2: -0.0107289, -0.0052244, -0.0107384, -0.0051171, -0.0043384, 0.0040841
3: 0.0011048, 0.0036102, 0.0010560, 0.0036145, -0.0018589, 0.0019747
4: -0.0015487, -0.0004833, -0.0015505, -0.0004625, -0.0008397, 0.0007905
5: -0.0145347, -0.0076115, -0.0145467, -0.0074766, -0.0054566, 0.0051367
6: 0.0034727, 0.0052299, 0.0034385, 0.0052329, -0.0013038, 0.0013849
7: 0.0058473, 0.0103937, 0.0057587, 0.0104016, -0.0033732, 0.0035832
8: 0.0035109, 0.0059018, 0.0034643, 0.0059059, -0.0017739, 0.0018844
9: -0.0087073, -0.0059349, -0.0087121, -0.0058809, -0.0021850, 0.0020570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017496, upper bound: 0.0017313
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017496, upper bound: 0.0017502
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0010930, 1.0056691, -0.0031462, 0.0036481
1: -0.0008787, 0.0001534, -0.0009916, 0.0001486, -0.0007839, 0.0009090
2: -0.0108669, -0.0053976, -0.0108418, -0.0047990, -0.0048173, 0.0041545
3: 0.0011836, 0.0036730, 0.0009112, 0.0036616, -0.0018910, 0.0021926
4: -0.0015754, -0.0005168, -0.0015705, -0.0004010, -0.0009324, 0.0008041
5: -0.0147082, -0.0078293, -0.0146766, -0.0070765, -0.0060589, 0.0052253
6: 0.0035280, 0.0052739, 0.0033369, 0.0052659, -0.0013262, 0.0015378
7: 0.0059904, 0.0105077, 0.0054960, 0.0104869, -0.0034314, 0.0039788
8: 0.0035861, 0.0059617, 0.0033261, 0.0059508, -0.0018045, 0.0020924
9: -0.0087768, -0.0060222, -0.0087641, -0.0057207, -0.0024263, 0.0020924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018127, upper bound: 0.0017125
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018127, upper bound: 0.0017284
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0014151, 1.0055836, 1.0010248, 1.0056758, -0.0032477, 0.0036919
1: -0.0009113, 0.0001273, -0.0010086, 0.0001503, -0.0008092, 0.0009199
2: -0.0107289, -0.0052244, -0.0108505, -0.0047090, -0.0048752, 0.0042885
3: 0.0011048, 0.0036102, 0.0008702, 0.0036655, -0.0019520, 0.0022190
4: -0.0015487, -0.0004833, -0.0015722, -0.0003835, -0.0009436, 0.0008300
5: -0.0145347, -0.0076115, -0.0146876, -0.0069632, -0.0061317, 0.0053939
6: 0.0034727, 0.0052299, 0.0033082, 0.0052687, -0.0013690, 0.0015563
7: 0.0058473, 0.0103937, 0.0054216, 0.0104941, -0.0035421, 0.0040266
8: 0.0035109, 0.0059018, 0.0032870, 0.0059546, -0.0018627, 0.0021175
9: -0.0087073, -0.0059349, -0.0087685, -0.0056753, -0.0024554, 0.0021599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018331, upper bound: 0.0017126
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018331, upper bound: 0.0017290
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0014017, 1.0055848, -0.0034160, 0.0033874
1: -0.0009555, 0.0001730, -0.0009147, 0.0001276, -0.0008512, 0.0008441
2: -0.0109709, -0.0049904, -0.0107304, -0.0052065, -0.0044731, 0.0045107
3: 0.0009983, 0.0037204, 0.0010967, 0.0036109, -0.0020531, 0.0020359
4: -0.0015955, -0.0004380, -0.0015490, -0.0004798, -0.0008658, 0.0008730
5: -0.0148391, -0.0073172, -0.0145366, -0.0075890, -0.0056259, 0.0056733
6: 0.0033980, 0.0053072, 0.0034670, 0.0052304, -0.0014399, 0.0014279
7: 0.0056541, 0.0105936, 0.0058325, 0.0103949, -0.0037256, 0.0036945
8: 0.0034093, 0.0060069, 0.0035031, 0.0059025, -0.0019592, 0.0019429
9: -0.0088292, -0.0058171, -0.0087080, -0.0059259, -0.0022529, 0.0022718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017126, upper bound: 0.0018127
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017126, upper bound: 0.0018332
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0011046, 1.0056679, 1.0013340, 1.0055909, -0.0034844, 0.0034264
1: -0.0009887, 0.0001483, -0.0009316, 0.0001291, -0.0008682, 0.0008538
2: -0.0108400, -0.0048142, -0.0107384, -0.0051171, -0.0045246, 0.0046011
3: 0.0009181, 0.0036608, 0.0010560, 0.0036145, -0.0020942, 0.0020594
4: -0.0015702, -0.0004039, -0.0015505, -0.0004625, -0.0008757, 0.0008905
5: -0.0146744, -0.0070956, -0.0145467, -0.0074766, -0.0056907, 0.0057870
6: 0.0033418, 0.0052654, 0.0034385, 0.0052329, -0.0014688, 0.0014444
7: 0.0055086, 0.0104855, 0.0057587, 0.0104016, -0.0038003, 0.0037370
8: 0.0033328, 0.0059501, 0.0034643, 0.0059059, -0.0019985, 0.0019653
9: -0.0087632, -0.0057283, -0.0087121, -0.0058809, -0.0022788, 0.0023174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018127
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018348
time: 1.67 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0010930, 1.0056691, -0.0031279, 0.0033489
1: -0.0009555, 0.0001730, -0.0009916, 0.0001486, -0.0007794, 0.0008344
2: -0.0109709, -0.0049904, -0.0108418, -0.0047990, -0.0044221, 0.0041304
3: 0.0009983, 0.0037204, 0.0009112, 0.0036616, -0.0018800, 0.0020128
4: -0.0015955, -0.0004380, -0.0015705, -0.0004010, -0.0008559, 0.0007994
5: -0.0148391, -0.0073172, -0.0146766, -0.0070765, -0.0055619, 0.0051949
6: 0.0033980, 0.0053072, 0.0033369, 0.0052659, -0.0013185, 0.0014117
7: 0.0056541, 0.0105936, 0.0054960, 0.0104869, -0.0034114, 0.0036524
8: 0.0034093, 0.0060069, 0.0033261, 0.0059508, -0.0017940, 0.0019208
9: -0.0088292, -0.0058171, -0.0087641, -0.0057207, -0.0022272, 0.0020803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018127
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018332
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0011046, 1.0056679, 1.0010248, 1.0056758, -0.0032061, 0.0034027
1: -0.0009887, 0.0001483, -0.0010086, 0.0001503, -0.0007989, 0.0008479
2: -0.0108400, -0.0048142, -0.0108505, -0.0047090, -0.0044932, 0.0042337
3: 0.0009181, 0.0036608, 0.0008702, 0.0036655, -0.0019270, 0.0020451
4: -0.0015702, -0.0004039, -0.0015722, -0.0003835, -0.0008697, 0.0008194
5: -0.0146744, -0.0070956, -0.0146876, -0.0069632, -0.0056513, 0.0053248
6: 0.0033418, 0.0052654, 0.0033082, 0.0052687, -0.0013515, 0.0014344
7: 0.0055086, 0.0104855, 0.0054216, 0.0104941, -0.0034967, 0.0037111
8: 0.0033328, 0.0059501, 0.0032870, 0.0059546, -0.0018389, 0.0019516
9: -0.0087632, -0.0057283, -0.0087685, -0.0056753, -0.0022630, 0.0021323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018127
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018348
time: 1.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.44 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017313
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017496
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017496, upper bound: 0.0017313
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017496, upper bound: 0.0017502
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0018127, upper bound: 0.0017125
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0018127, upper bound: 0.0017284
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0018331, upper bound: 0.0017126
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0018331, upper bound: 0.0017290
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017126, upper bound: 0.0018127
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017126, upper bound: 0.0018332
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018127
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018348
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018127
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018332
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018127
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018348

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0015463, 1.0056882, -0.0030621, 0.0030621
1: -0.0008787, 0.0001534, -0.0008787, 0.0001534, -0.0007630, 0.0007630
2: -0.0108669, -0.0053976, -0.0108669, -0.0053976, -0.0040435, 0.0040435
3: 0.0011836, 0.0036730, 0.0011836, 0.0036730, -0.0018404, 0.0018404
4: -0.0015754, -0.0005168, -0.0015754, -0.0005168, -0.0007826, 0.0007826
5: -0.0147082, -0.0078293, -0.0147082, -0.0078293, -0.0050856, 0.0050856
6: 0.0035280, 0.0052739, 0.0035280, 0.0052739, -0.0012908, 0.0012908
7: 0.0059904, 0.0105077, 0.0059904, 0.0105077, -0.0033397, 0.0033397
8: 0.0035861, 0.0059617, 0.0035861, 0.0059617, -0.0017563, 0.0017563
9: -0.0087768, -0.0060222, -0.0087768, -0.0060222, -0.0020365, 0.0020365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015997, upper bound: 0.0015998
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015924, upper bound: 0.0015924
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0014151, 1.0055836, -0.0030091, 0.0032570
1: -0.0008787, 0.0001534, -0.0009113, 0.0001273, -0.0007498, 0.0008116
2: -0.0108669, -0.0053976, -0.0107289, -0.0052244, -0.0043008, 0.0039735
3: 0.0011836, 0.0036730, 0.0011048, 0.0036102, -0.0018086, 0.0019576
4: -0.0015754, -0.0005168, -0.0015487, -0.0004833, -0.0008324, 0.0007691
5: -0.0147082, -0.0078293, -0.0145347, -0.0076115, -0.0054093, 0.0049977
6: 0.0035280, 0.0052739, 0.0034727, 0.0052299, -0.0012685, 0.0013729
7: 0.0059904, 0.0105077, 0.0058473, 0.0103937, -0.0032819, 0.0035522
8: 0.0035861, 0.0059617, 0.0035109, 0.0059018, -0.0017259, 0.0018681
9: -0.0087768, -0.0060222, -0.0087073, -0.0059349, -0.0021661, 0.0020013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015997, upper bound: 0.0016227
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015924, upper bound: 0.0016174
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0014151, 1.0055836, 1.0015463, 1.0056882, -0.0032570, 0.0030091
1: -0.0009113, 0.0001273, -0.0008787, 0.0001534, -0.0008116, 0.0007498
2: -0.0107289, -0.0052244, -0.0108669, -0.0053976, -0.0039735, 0.0043008
3: 0.0011048, 0.0036102, 0.0011836, 0.0036730, -0.0019576, 0.0018086
4: -0.0015487, -0.0004833, -0.0015754, -0.0005168, -0.0007691, 0.0008324
5: -0.0145347, -0.0076115, -0.0147082, -0.0078293, -0.0049977, 0.0054093
6: 0.0034727, 0.0052299, 0.0035280, 0.0052739, -0.0013729, 0.0012685
7: 0.0058473, 0.0103937, 0.0059904, 0.0105077, -0.0035522, 0.0032819
8: 0.0035109, 0.0059018, 0.0035861, 0.0059617, -0.0018681, 0.0017259
9: -0.0087073, -0.0059349, -0.0087768, -0.0060222, -0.0020013, 0.0021661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016226, upper bound: 0.0015998
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016174, upper bound: 0.0015924
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0014151, 1.0055836, 1.0014151, 1.0055836, -0.0030736, 0.0030736
1: -0.0009113, 0.0001273, -0.0009113, 0.0001273, -0.0007659, 0.0007659
2: -0.0107289, -0.0052244, -0.0107289, -0.0052244, -0.0040587, 0.0040587
3: 0.0011048, 0.0036102, 0.0011048, 0.0036102, -0.0018473, 0.0018473
4: -0.0015487, -0.0004833, -0.0015487, -0.0004833, -0.0007856, 0.0007856
5: -0.0145347, -0.0076115, -0.0145347, -0.0076115, -0.0051048, 0.0051048
6: 0.0034727, 0.0052299, 0.0034727, 0.0052299, -0.0012956, 0.0012956
7: 0.0058473, 0.0103937, 0.0058473, 0.0103937, -0.0033522, 0.0033522
8: 0.0035109, 0.0059018, 0.0035109, 0.0059018, -0.0017629, 0.0017629
9: -0.0087073, -0.0059349, -0.0087073, -0.0059349, -0.0020442, 0.0020442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016226, upper bound: 0.0016193
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016174, upper bound: 0.0016190
time: 1.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0012380, 1.0057670, -0.0032092, 0.0034676
1: -0.0008787, 0.0001534, -0.0009555, 0.0001730, -0.0007996, 0.0008640
2: -0.0108669, -0.0053976, -0.0109709, -0.0049904, -0.0045790, 0.0042377
3: 0.0011836, 0.0036730, 0.0009983, 0.0037204, -0.0019288, 0.0020842
4: -0.0015754, -0.0005168, -0.0015955, -0.0004380, -0.0008863, 0.0008202
5: -0.0147082, -0.0078293, -0.0148391, -0.0073172, -0.0057592, 0.0053299
6: 0.0035280, 0.0052739, 0.0033980, 0.0053072, -0.0013528, 0.0014617
7: 0.0059904, 0.0105077, 0.0056541, 0.0105936, -0.0035000, 0.0037820
8: 0.0035861, 0.0059617, 0.0034093, 0.0060069, -0.0018406, 0.0019889
9: -0.0087768, -0.0060222, -0.0088292, -0.0058171, -0.0023062, 0.0021343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016510, upper bound: 0.0015734
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015634
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0011046, 1.0056679, -0.0031448, 0.0036527
1: -0.0008787, 0.0001534, -0.0009887, 0.0001483, -0.0007836, 0.0009102
2: -0.0108669, -0.0053976, -0.0108400, -0.0048142, -0.0048234, 0.0041527
3: 0.0011836, 0.0036730, 0.0009181, 0.0036608, -0.0018901, 0.0021954
4: -0.0015754, -0.0005168, -0.0015702, -0.0004039, -0.0009336, 0.0008037
5: -0.0147082, -0.0078293, -0.0146744, -0.0070956, -0.0060665, 0.0052229
6: 0.0035280, 0.0052739, 0.0033418, 0.0052654, -0.0013256, 0.0015398
7: 0.0059904, 0.0105077, 0.0055086, 0.0104855, -0.0034298, 0.0039838
8: 0.0035861, 0.0059617, 0.0033328, 0.0059501, -0.0018037, 0.0020950
9: -0.0087768, -0.0060222, -0.0087632, -0.0057283, -0.0024293, 0.0020915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016510, upper bound: 0.0015952
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015879
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0014151, 1.0055836, 1.0012380, 1.0057670, -0.0034089, 0.0034147
1: -0.0009113, 0.0001273, -0.0009555, 0.0001730, -0.0008494, 0.0008508
2: -0.0107289, -0.0052244, -0.0109709, -0.0049904, -0.0045090, 0.0045014
3: 0.0011048, 0.0036102, 0.0009983, 0.0037204, -0.0020489, 0.0020523
4: -0.0015487, -0.0004833, -0.0015955, -0.0004380, -0.0008727, 0.0008712
5: -0.0145347, -0.0076115, -0.0148391, -0.0073172, -0.0056712, 0.0056616
6: 0.0034727, 0.0052299, 0.0033980, 0.0053072, -0.0014370, 0.0014394
7: 0.0058473, 0.0103937, 0.0056541, 0.0105936, -0.0037179, 0.0037242
8: 0.0035109, 0.0059018, 0.0034093, 0.0060069, -0.0019552, 0.0019585
9: -0.0087073, -0.0059349, -0.0088292, -0.0058171, -0.0022710, 0.0022672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016812, upper bound: 0.0015734
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016697, upper bound: 0.0015634
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0014151, 1.0055836, 1.0011046, 1.0056679, -0.0032267, 0.0034652
1: -0.0009113, 0.0001273, -0.0009887, 0.0001483, -0.0008040, 0.0008634
2: -0.0107289, -0.0052244, -0.0108400, -0.0048142, -0.0045757, 0.0042608
3: 0.0011048, 0.0036102, 0.0009181, 0.0036608, -0.0019393, 0.0020827
4: -0.0015487, -0.0004833, -0.0015702, -0.0004039, -0.0008856, 0.0008247
5: -0.0145347, -0.0076115, -0.0146744, -0.0070956, -0.0057551, 0.0053590
6: 0.0034727, 0.0052299, 0.0033418, 0.0052654, -0.0013602, 0.0014607
7: 0.0058473, 0.0103937, 0.0055086, 0.0104855, -0.0035192, 0.0037793
8: 0.0035109, 0.0059018, 0.0033328, 0.0059501, -0.0018507, 0.0019875
9: -0.0087073, -0.0059349, -0.0087632, -0.0057283, -0.0023046, 0.0021460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016812, upper bound: 0.0015937
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016697, upper bound: 0.0015928
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0015463, 1.0056882, -0.0034676, 0.0032092
1: -0.0009555, 0.0001730, -0.0008787, 0.0001534, -0.0008640, 0.0007996
2: -0.0109709, -0.0049904, -0.0108669, -0.0053976, -0.0042377, 0.0045790
3: 0.0009983, 0.0037204, 0.0011836, 0.0036730, -0.0020842, 0.0019288
4: -0.0015955, -0.0004380, -0.0015754, -0.0005168, -0.0008202, 0.0008863
5: -0.0148391, -0.0073172, -0.0147082, -0.0078293, -0.0053299, 0.0057592
6: 0.0033980, 0.0053072, 0.0035280, 0.0052739, -0.0014617, 0.0013528
7: 0.0056541, 0.0105936, 0.0059904, 0.0105077, -0.0037820, 0.0035000
8: 0.0034093, 0.0060069, 0.0035861, 0.0059617, -0.0019889, 0.0018406
9: -0.0088292, -0.0058171, -0.0087768, -0.0060222, -0.0021343, 0.0023062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016575
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0014151, 1.0055836, -0.0034147, 0.0034089
1: -0.0009555, 0.0001730, -0.0009113, 0.0001273, -0.0008508, 0.0008494
2: -0.0109709, -0.0049904, -0.0107289, -0.0052244, -0.0045014, 0.0045090
3: 0.0009983, 0.0037204, 0.0011048, 0.0036102, -0.0020523, 0.0020489
4: -0.0015955, -0.0004380, -0.0015487, -0.0004833, -0.0008712, 0.0008727
5: -0.0148391, -0.0073172, -0.0145347, -0.0076115, -0.0056616, 0.0056712
6: 0.0033980, 0.0053072, 0.0034727, 0.0052299, -0.0014394, 0.0014370
7: 0.0056541, 0.0105936, 0.0058473, 0.0103937, -0.0037242, 0.0037179
8: 0.0034093, 0.0060069, 0.0035109, 0.0059018, -0.0019585, 0.0019552
9: -0.0088292, -0.0058171, -0.0087073, -0.0059349, -0.0022672, 0.0022710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016849
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016697
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0011046, 1.0056679, 1.0015463, 1.0056882, -0.0036527, 0.0031448
1: -0.0009887, 0.0001483, -0.0008787, 0.0001534, -0.0009102, 0.0007836
2: -0.0108400, -0.0048142, -0.0108669, -0.0053976, -0.0041526, 0.0048234
3: 0.0009181, 0.0036608, 0.0011836, 0.0036730, -0.0021954, 0.0018901
4: -0.0015702, -0.0004039, -0.0015754, -0.0005168, -0.0008037, 0.0009336
5: -0.0146744, -0.0070956, -0.0147082, -0.0078293, -0.0052229, 0.0060665
6: 0.0033418, 0.0052654, 0.0035280, 0.0052739, -0.0015398, 0.0013256
7: 0.0055086, 0.0104855, 0.0059904, 0.0105077, -0.0039838, 0.0034298
8: 0.0033328, 0.0059501, 0.0035861, 0.0059617, -0.0020950, 0.0018037
9: -0.0087632, -0.0057283, -0.0087768, -0.0060222, -0.0020915, 0.0024293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016822, upper bound: 0.0017431
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016840, upper bound: 0.0017748
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0011046, 1.0056679, 1.0014151, 1.0055836, -0.0034652, 0.0032267
1: -0.0009887, 0.0001483, -0.0009113, 0.0001273, -0.0008634, 0.0008040
2: -0.0108400, -0.0048142, -0.0107289, -0.0052244, -0.0042608, 0.0045757
3: 0.0009181, 0.0036608, 0.0011048, 0.0036102, -0.0020827, 0.0019393
4: -0.0015702, -0.0004039, -0.0015487, -0.0004833, -0.0008247, 0.0008856
5: -0.0146744, -0.0070956, -0.0145347, -0.0076115, -0.0053590, 0.0057551
6: 0.0033418, 0.0052654, 0.0034727, 0.0052299, -0.0014607, 0.0013602
7: 0.0055086, 0.0104855, 0.0058473, 0.0103937, -0.0037793, 0.0035192
8: 0.0033328, 0.0059501, 0.0035109, 0.0059018, -0.0019875, 0.0018507
9: -0.0087632, -0.0057283, -0.0087073, -0.0059349, -0.0021460, 0.0023046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016822, upper bound: 0.0017501
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016840, upper bound: 0.0017784
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0012380, 1.0057670, -0.0031776, 0.0031776
1: -0.0009555, 0.0001730, -0.0009555, 0.0001730, -0.0007918, 0.0007918
2: -0.0109709, -0.0049904, -0.0109709, -0.0049904, -0.0041960, 0.0041960
3: 0.0009983, 0.0037204, 0.0009983, 0.0037204, -0.0019099, 0.0019099
4: -0.0015955, -0.0004380, -0.0015955, -0.0004380, -0.0008121, 0.0008121
5: -0.0148391, -0.0073172, -0.0148391, -0.0073172, -0.0052775, 0.0052775
6: 0.0033980, 0.0053072, 0.0033980, 0.0053072, -0.0013395, 0.0013395
7: 0.0056541, 0.0105936, 0.0056541, 0.0105936, -0.0034657, 0.0034657
8: 0.0034093, 0.0060069, 0.0034093, 0.0060069, -0.0018226, 0.0018226
9: -0.0088292, -0.0058171, -0.0088292, -0.0058171, -0.0021134, 0.0021134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016575
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0011046, 1.0056679, -0.0031266, 0.0033746
1: -0.0009555, 0.0001730, -0.0009887, 0.0001483, -0.0007791, 0.0008409
2: -0.0109709, -0.0049904, -0.0108400, -0.0048142, -0.0044561, 0.0041286
3: 0.0009983, 0.0037204, 0.0009181, 0.0036608, -0.0018792, 0.0020282
4: -0.0015955, -0.0004380, -0.0015702, -0.0004039, -0.0008625, 0.0007991
5: -0.0148391, -0.0073172, -0.0146744, -0.0070956, -0.0056046, 0.0051927
6: 0.0033980, 0.0053072, 0.0033418, 0.0052654, -0.0013180, 0.0014225
7: 0.0056541, 0.0105936, 0.0055086, 0.0104855, -0.0034100, 0.0036805
8: 0.0034093, 0.0060069, 0.0033328, 0.0059501, -0.0017933, 0.0019355
9: -0.0088292, -0.0058171, -0.0087632, -0.0057283, -0.0022443, 0.0020794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016849
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016697
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0011046, 1.0056679, 1.0012380, 1.0057670, -0.0033746, 0.0031266
1: -0.0009887, 0.0001483, -0.0009555, 0.0001730, -0.0008409, 0.0007791
2: -0.0108400, -0.0048142, -0.0109709, -0.0049904, -0.0041286, 0.0044561
3: 0.0009181, 0.0036608, 0.0009983, 0.0037204, -0.0020282, 0.0018792
4: -0.0015702, -0.0004039, -0.0015955, -0.0004380, -0.0007991, 0.0008625
5: -0.0146744, -0.0070956, -0.0148391, -0.0073172, -0.0051927, 0.0056046
6: 0.0033418, 0.0052654, 0.0033980, 0.0053072, -0.0014225, 0.0013180
7: 0.0055086, 0.0104855, 0.0056541, 0.0105936, -0.0036805, 0.0034100
8: 0.0033328, 0.0059501, 0.0034093, 0.0060069, -0.0019355, 0.0017933
9: -0.0087632, -0.0057283, -0.0088292, -0.0058171, -0.0020794, 0.0022443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016822, upper bound: 0.0017431
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016840, upper bound: 0.0017748
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0011046, 1.0056679, 1.0011046, 1.0056679, -0.0031867, 0.0031867
1: -0.0009887, 0.0001483, -0.0009887, 0.0001483, -0.0007941, 0.0007941
2: -0.0108400, -0.0048142, -0.0108400, -0.0048142, -0.0042081, 0.0042081
3: 0.0009181, 0.0036608, 0.0009181, 0.0036608, -0.0019153, 0.0019153
4: -0.0015702, -0.0004039, -0.0015702, -0.0004039, -0.0008145, 0.0008145
5: -0.0146744, -0.0070956, -0.0146744, -0.0070956, -0.0052926, 0.0052926
6: 0.0033418, 0.0052654, 0.0033418, 0.0052654, -0.0013433, 0.0013433
7: 0.0055086, 0.0104855, 0.0055086, 0.0104855, -0.0034756, 0.0034756
8: 0.0033328, 0.0059501, 0.0033328, 0.0059501, -0.0018278, 0.0018278
9: -0.0087632, -0.0057283, -0.0087632, -0.0057283, -0.0021194, 0.0021194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016822, upper bound: 0.0017502
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016840, upper bound: 0.0017784
time: 1.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.18 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015997, upper bound: 0.0015998
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015924, upper bound: 0.0015924
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015997, upper bound: 0.0016227
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015924, upper bound: 0.0016174
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016226, upper bound: 0.0015998
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016174, upper bound: 0.0015924
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016226, upper bound: 0.0016193
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016174, upper bound: 0.0016190
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016510, upper bound: 0.0015734
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015634
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016510, upper bound: 0.0015952
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015879
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016812, upper bound: 0.0015734
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016697, upper bound: 0.0015634
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016812, upper bound: 0.0015937
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016697, upper bound: 0.0015928
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016575
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016849
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016697
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016822, upper bound: 0.0017431
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016840, upper bound: 0.0017748
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016822, upper bound: 0.0017501
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016840, upper bound: 0.0017784
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016575
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016849
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016697
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016822, upper bound: 0.0017431
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016840, upper bound: 0.0017748
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016822, upper bound: 0.0017502
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -0.0016840, upper bound: 0.0017784

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0016177, 1.0056852, 1.0014151, 1.0055836, -0.0029288, 0.0032549
1: -0.0008609, 0.0001526, -0.0009113, 0.0001273, -0.0007298, 0.0008110
2: -0.0108629, -0.0054918, -0.0107289, -0.0052244, -0.0042980, 0.0038674
3: 0.0012265, 0.0036712, 0.0011048, 0.0036102, -0.0017603, 0.0019563
4: -0.0015746, -0.0005350, -0.0015487, -0.0004833, -0.0008319, 0.0007485
5: -0.0147032, -0.0079478, -0.0145347, -0.0076115, -0.0054058, 0.0048642
6: 0.0035581, 0.0052727, 0.0034727, 0.0052299, -0.0012346, 0.0013720
7: 0.0060682, 0.0105044, 0.0058473, 0.0103937, -0.0031943, 0.0035499
8: 0.0036270, 0.0059600, 0.0035109, 0.0059018, -0.0016798, 0.0018669
9: -0.0087748, -0.0060696, -0.0087073, -0.0059349, -0.0021647, 0.0019479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015919, upper bound: 0.0016166
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015919, upper bound: 0.0016174
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0015463, 1.0056882, -0.0031748, 0.0030066
1: -0.0008937, 0.0001266, -0.0008787, 0.0001534, -0.0007911, 0.0007492
2: -0.0107247, -0.0053178, -0.0108669, -0.0053976, -0.0039702, 0.0041923
3: 0.0011473, 0.0036083, 0.0011836, 0.0036730, -0.0019082, 0.0018070
4: -0.0015479, -0.0005014, -0.0015754, -0.0005168, -0.0007684, 0.0008114
5: -0.0145294, -0.0077290, -0.0147082, -0.0078293, -0.0049934, 0.0052728
6: 0.0035025, 0.0052286, 0.0035280, 0.0052739, -0.0013383, 0.0012674
7: 0.0059245, 0.0103903, 0.0059904, 0.0105077, -0.0034626, 0.0032791
8: 0.0035515, 0.0059000, 0.0035861, 0.0059617, -0.0018209, 0.0017245
9: -0.0087052, -0.0059820, -0.0087768, -0.0060222, -0.0019996, 0.0021115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016166, upper bound: 0.0015919
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016166, upper bound: 0.0015924
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0014151, 1.0055836, -0.0029929, 0.0030715
1: -0.0008937, 0.0001266, -0.0009113, 0.0001273, -0.0007458, 0.0007653
2: -0.0107247, -0.0053178, -0.0107289, -0.0052244, -0.0040559, 0.0039521
3: 0.0011473, 0.0036083, 0.0011048, 0.0036102, -0.0017988, 0.0018461
4: -0.0015479, -0.0005014, -0.0015487, -0.0004833, -0.0007850, 0.0007649
5: -0.0145294, -0.0077290, -0.0145347, -0.0076115, -0.0051012, 0.0049707
6: 0.0035025, 0.0052286, 0.0034727, 0.0052299, -0.0012616, 0.0012947
7: 0.0059245, 0.0103903, 0.0058473, 0.0103937, -0.0032642, 0.0033499
8: 0.0035515, 0.0059000, 0.0035109, 0.0059018, -0.0017166, 0.0017617
9: -0.0087052, -0.0059820, -0.0087073, -0.0059349, -0.0020428, 0.0019905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016313, upper bound: 0.0016148
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016313, upper bound: 0.0016190
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0016177, 1.0056852, 1.0012380, 1.0057670, -0.0031288, 0.0034655
1: -0.0008609, 0.0001526, -0.0009555, 0.0001730, -0.0007796, 0.0008635
2: -0.0108629, -0.0054918, -0.0109709, -0.0049904, -0.0045761, 0.0041316
3: 0.0012265, 0.0036712, 0.0009983, 0.0037204, -0.0018805, 0.0020829
4: -0.0015746, -0.0005350, -0.0015955, -0.0004380, -0.0008857, 0.0007997
5: -0.0147032, -0.0079478, -0.0148391, -0.0073172, -0.0057556, 0.0051964
6: 0.0035581, 0.0052727, 0.0033980, 0.0053072, -0.0013189, 0.0014608
7: 0.0060682, 0.0105044, 0.0056541, 0.0105936, -0.0034124, 0.0037796
8: 0.0036270, 0.0059600, 0.0034093, 0.0060069, -0.0017946, 0.0019877
9: -0.0087748, -0.0060696, -0.0088292, -0.0058171, -0.0023048, 0.0020809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015634
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015634
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0017924, 1.0059015, 1.0013114, 1.0057641, -0.0030799, 0.0036856
1: -0.0008173, 0.0002065, -0.0009372, 0.0001723, -0.0007674, 0.0009184
2: -0.0111484, -0.0057227, -0.0109671, -0.0050875, -0.0048668, 0.0040670
3: 0.0013316, 0.0038012, 0.0010425, 0.0037186, -0.0018511, 0.0022152
4: -0.0016299, -0.0005797, -0.0015948, -0.0004568, -0.0009420, 0.0007872
5: -0.0150624, -0.0082382, -0.0148343, -0.0074393, -0.0061212, 0.0051152
6: 0.0036318, 0.0053638, 0.0034290, 0.0053059, -0.0012983, 0.0015536
7: 0.0062589, 0.0107402, 0.0057342, 0.0105904, -0.0033591, 0.0040197
8: 0.0037273, 0.0060840, 0.0034514, 0.0060053, -0.0017665, 0.0021139
9: -0.0089186, -0.0061859, -0.0088272, -0.0058660, -0.0024512, 0.0020483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014355, upper bound: 0.0014654
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015194
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0016177, 1.0056852, 1.0011046, 1.0056679, -0.0030644, 0.0036506
1: -0.0008609, 0.0001526, -0.0009887, 0.0001483, -0.0007636, 0.0009096
2: -0.0108629, -0.0054918, -0.0108400, -0.0048142, -0.0048206, 0.0040466
3: 0.0012265, 0.0036712, 0.0009181, 0.0036608, -0.0018418, 0.0021941
4: -0.0015746, -0.0005350, -0.0015702, -0.0004039, -0.0009330, 0.0007832
5: -0.0147032, -0.0079478, -0.0146744, -0.0070956, -0.0060630, 0.0050895
6: 0.0035581, 0.0052727, 0.0033418, 0.0052654, -0.0012918, 0.0015389
7: 0.0060682, 0.0105044, 0.0055086, 0.0104855, -0.0033422, 0.0039815
8: 0.0036270, 0.0059600, 0.0033328, 0.0059501, -0.0017576, 0.0020938
9: -0.0087748, -0.0060696, -0.0087632, -0.0057283, -0.0024279, 0.0020381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014798, upper bound: 0.0015256
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016108, upper bound: 0.0015518
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0017924, 1.0059015, 1.0011774, 1.0056648, -0.0030152, 0.0038719
1: -0.0008173, 0.0002065, -0.0009706, 0.0001476, -0.0007513, 0.0009648
2: -0.0111484, -0.0057227, -0.0108360, -0.0049105, -0.0051128, 0.0039815
3: 0.0013316, 0.0038012, 0.0009619, 0.0036590, -0.0018122, 0.0023271
4: -0.0016299, -0.0005797, -0.0015694, -0.0004225, -0.0009896, 0.0007706
5: -0.0150624, -0.0082382, -0.0146694, -0.0072166, -0.0064305, 0.0050077
6: 0.0036318, 0.0053638, 0.0033725, 0.0052641, -0.0012710, 0.0016321
7: 0.0062589, 0.0107402, 0.0055880, 0.0104822, -0.0032885, 0.0042228
8: 0.0037273, 0.0060840, 0.0033745, 0.0059483, -0.0017294, 0.0022208
9: -0.0089186, -0.0061859, -0.0087612, -0.0057768, -0.0025751, 0.0020053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014370, upper bound: 0.0015020
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015442
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0012380, 1.0057670, -0.0033267, 0.0034121
1: -0.0008937, 0.0001266, -0.0009555, 0.0001730, -0.0008289, 0.0008502
2: -0.0107247, -0.0053178, -0.0109709, -0.0049904, -0.0045057, 0.0043929
3: 0.0011473, 0.0036083, 0.0009983, 0.0037204, -0.0019995, 0.0020508
4: -0.0015479, -0.0005014, -0.0015955, -0.0004380, -0.0008721, 0.0008502
5: -0.0145294, -0.0077290, -0.0148391, -0.0073172, -0.0056670, 0.0055251
6: 0.0035025, 0.0052286, 0.0033980, 0.0053072, -0.0014023, 0.0014383
7: 0.0059245, 0.0103903, 0.0056541, 0.0105936, -0.0036283, 0.0037214
8: 0.0035515, 0.0059000, 0.0034093, 0.0060069, -0.0019081, 0.0019571
9: -0.0087052, -0.0059820, -0.0088292, -0.0058171, -0.0022693, 0.0022125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016697, upper bound: 0.0015634
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016697, upper bound: 0.0015634
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0016536, 1.0057963, 1.0013114, 1.0057641, -0.0032774, 0.0036307
1: -0.0008519, 0.0001803, -0.0009372, 0.0001723, -0.0008166, 0.0009047
2: -0.0110097, -0.0055392, -0.0109671, -0.0050875, -0.0047943, 0.0043278
3: 0.0012481, 0.0037380, 0.0010425, 0.0037186, -0.0019698, 0.0021822
4: -0.0016030, -0.0005442, -0.0015948, -0.0004568, -0.0009279, 0.0008376
5: -0.0148878, -0.0080074, -0.0148343, -0.0074393, -0.0060300, 0.0054432
6: 0.0035732, 0.0053195, 0.0034290, 0.0053059, -0.0013815, 0.0015305
7: 0.0061073, 0.0106256, 0.0057342, 0.0105904, -0.0035745, 0.0039598
8: 0.0036476, 0.0060237, 0.0034514, 0.0060053, -0.0018798, 0.0020824
9: -0.0088487, -0.0060935, -0.0088272, -0.0058660, -0.0024147, 0.0021797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014768, upper bound: 0.0014655
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016312, upper bound: 0.0015194
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0011046, 1.0056679, -0.0031460, 0.0034630
1: -0.0008937, 0.0001266, -0.0009887, 0.0001483, -0.0007839, 0.0008629
2: -0.0107247, -0.0053178, -0.0108400, -0.0048142, -0.0045729, 0.0041542
3: 0.0011473, 0.0036083, 0.0009181, 0.0036608, -0.0018908, 0.0020814
4: -0.0015479, -0.0005014, -0.0015702, -0.0004039, -0.0008851, 0.0008040
5: -0.0145294, -0.0077290, -0.0146744, -0.0070956, -0.0057515, 0.0052249
6: 0.0035025, 0.0052286, 0.0033418, 0.0052654, -0.0013261, 0.0014598
7: 0.0059245, 0.0103903, 0.0055086, 0.0104855, -0.0034311, 0.0037769
8: 0.0035515, 0.0059000, 0.0033328, 0.0059501, -0.0018044, 0.0019862
9: -0.0087052, -0.0059820, -0.0087632, -0.0057283, -0.0023031, 0.0020923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015394
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016635, upper bound: 0.0015508
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0016536, 1.0057963, 1.0011774, 1.0056648, -0.0030894, 0.0036856
1: -0.0008519, 0.0001803, -0.0009706, 0.0001476, -0.0007698, 0.0009184
2: -0.0110097, -0.0055392, -0.0108360, -0.0049105, -0.0048668, 0.0040796
3: 0.0012481, 0.0037380, 0.0009619, 0.0036590, -0.0018568, 0.0022151
4: -0.0016030, -0.0005442, -0.0015694, -0.0004225, -0.0009420, 0.0007896
5: -0.0148878, -0.0080074, -0.0146694, -0.0072166, -0.0061211, 0.0051310
6: 0.0035732, 0.0053195, 0.0033725, 0.0052641, -0.0013023, 0.0015536
7: 0.0061073, 0.0106256, 0.0055880, 0.0104822, -0.0033695, 0.0040197
8: 0.0036476, 0.0060237, 0.0033745, 0.0059483, -0.0017720, 0.0021139
9: -0.0088487, -0.0060935, -0.0087612, -0.0057768, -0.0024512, 0.0020547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015309
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016605, upper bound: 0.0015499
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0015463, 1.0056882, -0.0033875, 0.0032071
1: -0.0009376, 0.0001723, -0.0008787, 0.0001534, -0.0008441, 0.0007991
2: -0.0109673, -0.0050852, -0.0108669, -0.0053976, -0.0042350, 0.0044732
3: 0.0010414, 0.0037187, 0.0011836, 0.0036730, -0.0020360, 0.0019276
4: -0.0015948, -0.0004563, -0.0015754, -0.0005168, -0.0008197, 0.0008658
5: -0.0148345, -0.0074363, -0.0147082, -0.0078293, -0.0053265, 0.0056261
6: 0.0034283, 0.0053060, 0.0035280, 0.0052739, -0.0014280, 0.0013519
7: 0.0057323, 0.0105906, 0.0059904, 0.0105077, -0.0036946, 0.0034978
8: 0.0034504, 0.0060053, 0.0035861, 0.0059617, -0.0019430, 0.0018395
9: -0.0088273, -0.0058648, -0.0087768, -0.0060222, -0.0021330, 0.0022530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0014869, 1.0059855, 1.0016201, 1.0056851, -0.0033364, 0.0034239
1: -0.0008935, 0.0002275, -0.0008603, 0.0001526, -0.0008314, 0.0008532
2: -0.0112595, -0.0053190, -0.0108627, -0.0054950, -0.0045213, 0.0044057
3: 0.0011479, 0.0038517, 0.0012279, 0.0036711, -0.0020053, 0.0020579
4: -0.0016514, -0.0005016, -0.0015746, -0.0005357, -0.0008751, 0.0008527
5: -0.0152021, -0.0077305, -0.0147029, -0.0079518, -0.0056866, 0.0055412
6: 0.0035029, 0.0053993, 0.0035591, 0.0052726, -0.0014064, 0.0014433
7: 0.0059255, 0.0108320, 0.0060708, 0.0105042, -0.0036389, 0.0037343
8: 0.0035520, 0.0061323, 0.0036284, 0.0059599, -0.0019136, 0.0019638
9: -0.0089745, -0.0059826, -0.0087746, -0.0060712, -0.0022771, 0.0022190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014076, upper bound: 0.0015207
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0015963
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0014151, 1.0055836, -0.0033346, 0.0034069
1: -0.0009376, 0.0001723, -0.0009113, 0.0001273, -0.0008309, 0.0008489
2: -0.0109673, -0.0050852, -0.0107289, -0.0052244, -0.0044987, 0.0044033
3: 0.0010414, 0.0037187, 0.0011048, 0.0036102, -0.0020042, 0.0020476
4: -0.0015948, -0.0004563, -0.0015487, -0.0004833, -0.0008707, 0.0008522
5: -0.0148345, -0.0074363, -0.0145347, -0.0076115, -0.0056582, 0.0055382
6: 0.0034283, 0.0053060, 0.0034727, 0.0052299, -0.0014056, 0.0014361
7: 0.0057323, 0.0105906, 0.0058473, 0.0103937, -0.0036368, 0.0037157
8: 0.0034504, 0.0060053, 0.0035109, 0.0059018, -0.0019126, 0.0019540
9: -0.0088273, -0.0058648, -0.0087073, -0.0059349, -0.0022658, 0.0022177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016697
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016697
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0014869, 1.0059855, 1.0014881, 1.0055804, -0.0032832, 0.0036260
1: -0.0008935, 0.0002275, -0.0008932, 0.0001265, -0.0008181, 0.0009035
2: -0.0112595, -0.0053190, -0.0107246, -0.0053207, -0.0047881, 0.0043354
3: 0.0011479, 0.0038517, 0.0011486, 0.0036083, -0.0019733, 0.0021793
4: -0.0016514, -0.0005016, -0.0015478, -0.0005019, -0.0009267, 0.0008391
5: -0.0152021, -0.0077305, -0.0145293, -0.0077326, -0.0060222, 0.0054528
6: 0.0035029, 0.0053993, 0.0035035, 0.0052285, -0.0013840, 0.0015285
7: 0.0059255, 0.0108320, 0.0059269, 0.0103901, -0.0035808, 0.0039547
8: 0.0035520, 0.0061323, 0.0035527, 0.0058999, -0.0018831, 0.0020797
9: -0.0089745, -0.0059826, -0.0087051, -0.0059834, -0.0024115, 0.0021835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0015672
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0016312
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0010141, 1.0054914, 1.0015520, 1.0056291, -0.0036182, 0.0029657
1: -0.0010113, 0.0001044, -0.0008772, 0.0001386, -0.0009016, 0.0007390
2: -0.0106071, -0.0046948, -0.0107887, -0.0054051, -0.0039161, 0.0047778
3: 0.0008637, 0.0035548, 0.0011870, 0.0036374, -0.0021747, 0.0017824
4: -0.0015251, -0.0003808, -0.0015602, -0.0005183, -0.0007580, 0.0009247
5: -0.0143814, -0.0069453, -0.0146099, -0.0078388, -0.0049254, 0.0060092
6: 0.0033036, 0.0051910, 0.0035304, 0.0052490, -0.0015252, 0.0012501
7: 0.0054099, 0.0102931, 0.0059966, 0.0104431, -0.0039462, 0.0032345
8: 0.0032809, 0.0058489, 0.0035894, 0.0059278, -0.0020753, 0.0017010
9: -0.0086459, -0.0056682, -0.0087374, -0.0060259, -0.0019724, 0.0024064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015256, upper bound: 0.0014798
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0014370
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0011091, 1.0056273, 1.0015463, 1.0056882, -0.0036484, 0.0030886
1: -0.0009876, 0.0001382, -0.0008787, 0.0001534, -0.0009091, 0.0007696
2: -0.0107864, -0.0048204, -0.0108669, -0.0053976, -0.0040785, 0.0048177
3: 0.0009209, 0.0036364, 0.0011836, 0.0036730, -0.0021928, 0.0018563
4: -0.0015598, -0.0004051, -0.0015754, -0.0005168, -0.0007894, 0.0009324
5: -0.0146071, -0.0071033, -0.0147082, -0.0078293, -0.0051296, 0.0060593
6: 0.0033437, 0.0052483, 0.0035280, 0.0052739, -0.0015379, 0.0013020
7: 0.0055136, 0.0104412, 0.0059904, 0.0105077, -0.0039791, 0.0033686
8: 0.0033354, 0.0059268, 0.0035861, 0.0059617, -0.0020926, 0.0017715
9: -0.0087363, -0.0057314, -0.0087768, -0.0060222, -0.0020541, 0.0024264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015518, upper bound: 0.0016108
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015442, upper bound: 0.0015963
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0010141, 1.0054914, 1.0014273, 1.0055231, -0.0034423, 0.0030501
1: -0.0010113, 0.0001044, -0.0009083, 0.0001122, -0.0008577, 0.0007600
2: -0.0106071, -0.0046948, -0.0106489, -0.0052404, -0.0040277, 0.0045455
3: 0.0008637, 0.0035548, 0.0011121, 0.0035738, -0.0020689, 0.0018332
4: -0.0015251, -0.0003808, -0.0015332, -0.0004864, -0.0007795, 0.0008798
5: -0.0143814, -0.0069453, -0.0144340, -0.0076316, -0.0050658, 0.0057171
6: 0.0033036, 0.0051910, 0.0034778, 0.0052043, -0.0014511, 0.0012857
7: 0.0054099, 0.0102931, 0.0058605, 0.0103276, -0.0037543, 0.0033266
8: 0.0032809, 0.0058489, 0.0035178, 0.0058670, -0.0019744, 0.0017494
9: -0.0086459, -0.0056682, -0.0086670, -0.0059430, -0.0020286, 0.0022894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015522, upper bound: 0.0015727
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015430, upper bound: 0.0015532
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0011091, 1.0056273, 1.0014151, 1.0055836, -0.0034612, 0.0031651
1: -0.0009876, 0.0001382, -0.0009113, 0.0001273, -0.0008624, 0.0007887
2: -0.0107864, -0.0048204, -0.0107289, -0.0052244, -0.0041795, 0.0045705
3: 0.0009209, 0.0036364, 0.0011048, 0.0036102, -0.0020803, 0.0019023
4: -0.0015598, -0.0004051, -0.0015487, -0.0004833, -0.0008089, 0.0008846
5: -0.0146071, -0.0071033, -0.0145347, -0.0076115, -0.0052567, 0.0057484
6: 0.0033437, 0.0052483, 0.0034727, 0.0052299, -0.0014590, 0.0013342
7: 0.0055136, 0.0104412, 0.0058473, 0.0103937, -0.0037749, 0.0034520
8: 0.0033354, 0.0059268, 0.0035109, 0.0059018, -0.0019852, 0.0018154
9: -0.0087363, -0.0057314, -0.0087073, -0.0059349, -0.0021050, 0.0023019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015645, upper bound: 0.0016471
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016445
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0012380, 1.0057670, -0.0030972, 0.0031756
1: -0.0009376, 0.0001723, -0.0009555, 0.0001730, -0.0007717, 0.0007913
2: -0.0109673, -0.0050852, -0.0109709, -0.0049904, -0.0041933, 0.0040898
3: 0.0010414, 0.0037187, 0.0009983, 0.0037204, -0.0018615, 0.0019086
4: -0.0015948, -0.0004563, -0.0015955, -0.0004380, -0.0008116, 0.0007916
5: -0.0148345, -0.0074363, -0.0148391, -0.0073172, -0.0052741, 0.0051439
6: 0.0034283, 0.0053060, 0.0033980, 0.0053072, -0.0013056, 0.0013386
7: 0.0057323, 0.0105906, 0.0056541, 0.0105936, -0.0033779, 0.0034634
8: 0.0034504, 0.0060053, 0.0034093, 0.0060069, -0.0017764, 0.0018214
9: -0.0088273, -0.0058648, -0.0088292, -0.0058171, -0.0021120, 0.0020599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0014869, 1.0059855, 1.0013114, 1.0057641, -0.0030516, 0.0033998
1: -0.0008935, 0.0002275, -0.0009372, 0.0001723, -0.0007604, 0.0008471
2: -0.0112595, -0.0053190, -0.0109671, -0.0050875, -0.0044894, 0.0040296
3: 0.0011479, 0.0038517, 0.0010425, 0.0037186, -0.0018341, 0.0020434
4: -0.0016514, -0.0005016, -0.0015948, -0.0004568, -0.0008689, 0.0007799
5: -0.0152021, -0.0077305, -0.0148343, -0.0074393, -0.0056464, 0.0050682
6: 0.0035029, 0.0053993, 0.0034290, 0.0053059, -0.0012864, 0.0014331
7: 0.0059255, 0.0108320, 0.0057342, 0.0105904, -0.0033282, 0.0037079
8: 0.0035520, 0.0061323, 0.0034514, 0.0060053, -0.0017503, 0.0019500
9: -0.0089745, -0.0059826, -0.0088272, -0.0058660, -0.0022611, 0.0020295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014067, upper bound: 0.0015189
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0015963
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0011046, 1.0056679, -0.0030462, 0.0033726
1: -0.0009376, 0.0001723, -0.0009887, 0.0001483, -0.0007590, 0.0008404
2: -0.0109673, -0.0050852, -0.0108400, -0.0048142, -0.0044534, 0.0040224
3: 0.0010414, 0.0037187, 0.0009181, 0.0036608, -0.0018308, 0.0020270
4: -0.0015948, -0.0004563, -0.0015702, -0.0004039, -0.0008620, 0.0007785
5: -0.0148345, -0.0074363, -0.0146744, -0.0070956, -0.0056012, 0.0050592
6: 0.0034283, 0.0053060, 0.0033418, 0.0052654, -0.0012841, 0.0014217
7: 0.0057323, 0.0105906, 0.0055086, 0.0104855, -0.0033223, 0.0036783
8: 0.0034504, 0.0060053, 0.0033328, 0.0059501, -0.0017472, 0.0019344
9: -0.0088273, -0.0058648, -0.0087632, -0.0057283, -0.0022430, 0.0020259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014482, upper bound: 0.0016108
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016475
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0014869, 1.0059855, 1.0011774, 1.0056648, -0.0030002, 0.0035987
1: -0.0008935, 0.0002275, -0.0009706, 0.0001476, -0.0007476, 0.0008967
2: -0.0112595, -0.0053190, -0.0108360, -0.0049105, -0.0047520, 0.0039617
3: 0.0011479, 0.0038517, 0.0009619, 0.0036590, -0.0018032, 0.0021629
4: -0.0016514, -0.0005016, -0.0015694, -0.0004225, -0.0009197, 0.0007668
5: -0.0152021, -0.0077305, -0.0146694, -0.0072166, -0.0059768, 0.0049828
6: 0.0035029, 0.0053993, 0.0033725, 0.0052641, -0.0012647, 0.0015170
7: 0.0059255, 0.0108320, 0.0055880, 0.0104822, -0.0032722, 0.0039249
8: 0.0035520, 0.0061323, 0.0033745, 0.0059483, -0.0017208, 0.0020640
9: -0.0089745, -0.0059826, -0.0087612, -0.0057768, -0.0023934, 0.0019953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014078, upper bound: 0.0015668
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0016312
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0010141, 1.0054914, 1.0012436, 1.0057005, -0.0033953, 0.0029681
1: -0.0010113, 0.0001044, -0.0009541, 0.0001565, -0.0008460, 0.0007396
2: -0.0106071, -0.0046948, -0.0108832, -0.0049979, -0.0039194, 0.0044835
3: 0.0008637, 0.0035548, 0.0010017, 0.0036804, -0.0020407, 0.0017839
4: -0.0015251, -0.0003808, -0.0015785, -0.0004394, -0.0007586, 0.0008678
5: -0.0143814, -0.0069453, -0.0147288, -0.0073265, -0.0049296, 0.0056390
6: 0.0033036, 0.0051910, 0.0034004, 0.0052792, -0.0014312, 0.0012512
7: 0.0054099, 0.0102931, 0.0056602, 0.0105212, -0.0037031, 0.0032372
8: 0.0032809, 0.0058489, 0.0034125, 0.0059688, -0.0019474, 0.0017024
9: -0.0086459, -0.0056682, -0.0087850, -0.0058208, -0.0019740, 0.0022581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015256, upper bound: 0.0014788
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0014333
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0011091, 1.0056273, 1.0012380, 1.0057670, -0.0033699, 0.0031045
1: -0.0009876, 0.0001382, -0.0009555, 0.0001730, -0.0008397, 0.0007735
2: -0.0107864, -0.0048204, -0.0109709, -0.0049904, -0.0040994, 0.0044499
3: 0.0009209, 0.0036364, 0.0009983, 0.0037204, -0.0020254, 0.0018659
4: -0.0015598, -0.0004051, -0.0015955, -0.0004380, -0.0007934, 0.0008613
5: -0.0146071, -0.0071033, -0.0148391, -0.0073172, -0.0051560, 0.0055968
6: 0.0033437, 0.0052483, 0.0033980, 0.0053072, -0.0014205, 0.0013086
7: 0.0055136, 0.0104412, 0.0056541, 0.0105936, -0.0036753, 0.0033858
8: 0.0033354, 0.0059268, 0.0034093, 0.0060069, -0.0019328, 0.0017806
9: -0.0087363, -0.0057314, -0.0088292, -0.0058171, -0.0020647, 0.0022412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015518, upper bound: 0.0016108
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015442, upper bound: 0.0015963
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0010141, 1.0054914, 1.0011164, 1.0055988, -0.0032109, 0.0030355
1: -0.0010113, 0.0001044, -0.0009858, 0.0001311, -0.0008001, 0.0007564
2: -0.0106071, -0.0046948, -0.0107488, -0.0048298, -0.0040083, 0.0042400
3: 0.0008637, 0.0035548, 0.0009252, 0.0036192, -0.0019299, 0.0018244
4: -0.0015251, -0.0003808, -0.0015525, -0.0004069, -0.0007758, 0.0008206
5: -0.0143814, -0.0069453, -0.0145597, -0.0071152, -0.0050414, 0.0053328
6: 0.0033036, 0.0051910, 0.0033467, 0.0052362, -0.0013535, 0.0012796
7: 0.0054099, 0.0102931, 0.0055214, 0.0104101, -0.0035020, 0.0033106
8: 0.0032809, 0.0058489, 0.0033395, 0.0059104, -0.0018417, 0.0017410
9: -0.0086459, -0.0056682, -0.0087173, -0.0057362, -0.0020188, 0.0021355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015522, upper bound: 0.0015727
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015430, upper bound: 0.0015531
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0011091, 1.0056273, 1.0011046, 1.0056679, -0.0031827, 0.0031645
1: -0.0009876, 0.0001382, -0.0009887, 0.0001483, -0.0007930, 0.0007885
2: -0.0107864, -0.0048204, -0.0108400, -0.0048142, -0.0041786, 0.0042027
3: 0.0009209, 0.0036364, 0.0009181, 0.0036608, -0.0019129, 0.0019019
4: -0.0015598, -0.0004051, -0.0015702, -0.0004039, -0.0008088, 0.0008134
5: -0.0146071, -0.0071033, -0.0146744, -0.0070956, -0.0052556, 0.0052859
6: 0.0033437, 0.0052483, 0.0033418, 0.0052654, -0.0013416, 0.0013339
7: 0.0055136, 0.0104412, 0.0055086, 0.0104855, -0.0034712, 0.0034513
8: 0.0033354, 0.0059268, 0.0033328, 0.0059501, -0.0018255, 0.0018150
9: -0.0087363, -0.0057314, -0.0087632, -0.0057283, -0.0021046, 0.0021167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016737, upper bound: 0.0017759
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016737, upper bound: 0.0017784
time: 1.11 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.53 seconds
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015919, upper bound: 0.0016166
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015919, upper bound: 0.0016174
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016166, upper bound: 0.0015919
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016166, upper bound: 0.0015924
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016313, upper bound: 0.0016148
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016313, upper bound: 0.0016190
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015634
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015634
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014355, upper bound: 0.0014654
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015194
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014798, upper bound: 0.0015256
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016108, upper bound: 0.0015518
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014370, upper bound: 0.0015020
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015442
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016697, upper bound: 0.0015634
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016697, upper bound: 0.0015634
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014768, upper bound: 0.0014655
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016312, upper bound: 0.0015194
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015394
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016635, upper bound: 0.0015508
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015309
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016605, upper bound: 0.0015499
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014076, upper bound: 0.0015207
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0015963
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016697
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016697
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0015672
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0016312
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015256, upper bound: 0.0014798
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0014370
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015518, upper bound: 0.0016108
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015442, upper bound: 0.0015963
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015522, upper bound: 0.0015727
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015430, upper bound: 0.0015532
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015645, upper bound: 0.0016471
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016445
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014067, upper bound: 0.0015189
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0015963
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014482, upper bound: 0.0016108
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016475
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0014078, upper bound: 0.0015668
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0016312
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015256, upper bound: 0.0014788
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0014333
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015518, upper bound: 0.0016108
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015442, upper bound: 0.0015963
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015522, upper bound: 0.0015727
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0015430, upper bound: 0.0015531
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016737, upper bound: 0.0017759
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -0.0016737, upper bound: 0.0017784

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0014859, 1.0055805, -0.0029908, 0.0029908
1: -0.0008937, 0.0001266, -0.0008937, 0.0001266, -0.0007452, 0.0007452
2: -0.0107247, -0.0053178, -0.0107247, -0.0053178, -0.0039493, 0.0039493
3: 0.0011473, 0.0036083, 0.0011473, 0.0036083, -0.0017975, 0.0017975
4: -0.0015479, -0.0005014, -0.0015479, -0.0005014, -0.0007644, 0.0007644
5: -0.0145294, -0.0077290, -0.0145294, -0.0077290, -0.0049672, 0.0049672
6: 0.0035025, 0.0052286, 0.0035025, 0.0052286, -0.0012607, 0.0012607
7: 0.0059245, 0.0103903, 0.0059245, 0.0103903, -0.0032619, 0.0032619
8: 0.0035515, 0.0059000, 0.0035515, 0.0059000, -0.0017154, 0.0017154
9: -0.0087052, -0.0059820, -0.0087052, -0.0059820, -0.0019891, 0.0019891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015843, upper bound: 0.0015475
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015915, upper bound: 0.0015765
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0016536, 1.0057963, -0.0032747, 0.0028948
1: -0.0008937, 0.0001266, -0.0008519, 0.0001803, -0.0008160, 0.0007213
2: -0.0107247, -0.0053178, -0.0110097, -0.0055392, -0.0038226, 0.0043242
3: 0.0011473, 0.0036083, 0.0012481, 0.0037380, -0.0019682, 0.0017399
4: -0.0015479, -0.0005014, -0.0016030, -0.0005442, -0.0007399, 0.0008369
5: -0.0145294, -0.0077290, -0.0148878, -0.0080074, -0.0048078, 0.0054387
6: 0.0035025, 0.0052286, 0.0035732, 0.0053195, -0.0013804, 0.0012203
7: 0.0059245, 0.0103903, 0.0061073, 0.0106256, -0.0035715, 0.0031572
8: 0.0035515, 0.0059000, 0.0036476, 0.0060237, -0.0018782, 0.0016604
9: -0.0087052, -0.0059820, -0.0088487, -0.0060935, -0.0019253, 0.0021779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015843, upper bound: 0.0015475
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015915, upper bound: 0.0015781
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0016177, 1.0056852, 1.0013096, 1.0057642, -0.0031268, 0.0033854
1: -0.0008609, 0.0001526, -0.0009376, 0.0001723, -0.0007791, 0.0008435
2: -0.0108629, -0.0054918, -0.0109673, -0.0050852, -0.0044704, 0.0041289
3: 0.0012265, 0.0036712, 0.0010414, 0.0037187, -0.0018793, 0.0020347
4: -0.0015746, -0.0005350, -0.0015948, -0.0004563, -0.0008652, 0.0007991
5: -0.0147032, -0.0079478, -0.0148345, -0.0074363, -0.0056225, 0.0051931
6: 0.0035581, 0.0052727, 0.0034283, 0.0053060, -0.0013181, 0.0014271
7: 0.0060682, 0.0105044, 0.0057323, 0.0105906, -0.0034102, 0.0036922
8: 0.0036270, 0.0059600, 0.0034504, 0.0060053, -0.0017934, 0.0019417
9: -0.0087748, -0.0060696, -0.0088273, -0.0058648, -0.0022515, 0.0020795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015505, upper bound: 0.0014484
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016108, upper bound: 0.0015312
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0016177, 1.0056852, 1.0014869, 1.0059855, -0.0034013, 0.0032741
1: -0.0008609, 0.0001526, -0.0008935, 0.0002275, -0.0008475, 0.0008158
2: -0.0108629, -0.0054918, -0.0112595, -0.0053190, -0.0043235, 0.0044914
3: 0.0012265, 0.0036712, 0.0011479, 0.0038517, -0.0020443, 0.0019679
4: -0.0015746, -0.0005350, -0.0016514, -0.0005016, -0.0008368, 0.0008693
5: -0.0147032, -0.0079478, -0.0152021, -0.0077305, -0.0054378, 0.0056490
6: 0.0035581, 0.0052727, 0.0035029, 0.0053993, -0.0014338, 0.0013802
7: 0.0060682, 0.0105044, 0.0059255, 0.0108320, -0.0037096, 0.0035709
8: 0.0036270, 0.0059600, 0.0035520, 0.0061323, -0.0019509, 0.0018779
9: -0.0087748, -0.0060696, -0.0089745, -0.0059826, -0.0021775, 0.0022621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015505, upper bound: 0.0014484
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0016108, upper bound: 0.0015312
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0013096, 1.0057642, -0.0033247, 0.0033320
1: -0.0008937, 0.0001266, -0.0009376, 0.0001723, -0.0008284, 0.0008303
2: -0.0107247, -0.0053178, -0.0109673, -0.0050852, -0.0043999, 0.0043902
3: 0.0011473, 0.0036083, 0.0010414, 0.0037187, -0.0019982, 0.0020026
4: -0.0015479, -0.0005014, -0.0015948, -0.0004563, -0.0008516, 0.0008497
5: -0.0145294, -0.0077290, -0.0148345, -0.0074363, -0.0055339, 0.0055217
6: 0.0035025, 0.0052286, 0.0034283, 0.0053060, -0.0014015, 0.0014046
7: 0.0059245, 0.0103903, 0.0057323, 0.0105906, -0.0036260, 0.0036341
8: 0.0035515, 0.0059000, 0.0034504, 0.0060053, -0.0019069, 0.0019111
9: -0.0087052, -0.0059820, -0.0088273, -0.0058648, -0.0022160, 0.0022111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015893, upper bound: 0.0014484
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016423, upper bound: 0.0015312
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0014869, 1.0059855, -0.0036007, 0.0032208
1: -0.0008937, 0.0001266, -0.0008935, 0.0002275, -0.0008972, 0.0008025
2: -0.0107247, -0.0053178, -0.0112595, -0.0053190, -0.0042530, 0.0047547
3: 0.0011473, 0.0036083, 0.0011479, 0.0038517, -0.0021642, 0.0019358
4: -0.0015479, -0.0005014, -0.0016514, -0.0005016, -0.0008232, 0.0009203
5: -0.0145294, -0.0077290, -0.0152021, -0.0077305, -0.0053492, 0.0059802
6: 0.0035025, 0.0052286, 0.0035029, 0.0053993, -0.0015178, 0.0013577
7: 0.0059245, 0.0103903, 0.0059255, 0.0108320, -0.0039271, 0.0035127
8: 0.0035515, 0.0059000, 0.0035520, 0.0061323, -0.0020652, 0.0018473
9: -0.0087052, -0.0059820, -0.0089745, -0.0059826, -0.0021420, 0.0023947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015893, upper bound: 0.0014484
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016423, upper bound: 0.0015312
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0016536, 1.0057963, 1.0013139, 1.0057251, -0.0032223, 0.0036269
1: -0.0008519, 0.0001803, -0.0009366, 0.0001626, -0.0008029, 0.0009037
2: -0.0110097, -0.0055392, -0.0109157, -0.0050907, -0.0047893, 0.0042551
3: 0.0012481, 0.0037380, 0.0010439, 0.0036952, -0.0019367, 0.0021799
4: -0.0016030, -0.0005442, -0.0015848, -0.0004574, -0.0009270, 0.0008236
5: -0.0148878, -0.0080074, -0.0147696, -0.0074433, -0.0060236, 0.0053518
6: 0.0035732, 0.0053195, 0.0034300, 0.0052895, -0.0013583, 0.0015289
7: 0.0061073, 0.0106256, 0.0057369, 0.0105479, -0.0035144, 0.0039556
8: 0.0036476, 0.0060237, 0.0034528, 0.0059829, -0.0018482, 0.0020802
9: -0.0088487, -0.0060935, -0.0088013, -0.0058676, -0.0024121, 0.0021431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015672, upper bound: 0.0014084
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015672, upper bound: 0.0015194
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0011091, 1.0056273, -0.0030861, 0.0034590
1: -0.0008937, 0.0001266, -0.0009876, 0.0001382, -0.0007690, 0.0008619
2: -0.0107247, -0.0053178, -0.0107864, -0.0048204, -0.0045676, 0.0040752
3: 0.0011473, 0.0036083, 0.0009209, 0.0036364, -0.0018549, 0.0020790
4: -0.0015479, -0.0005014, -0.0015598, -0.0004051, -0.0008841, 0.0007887
5: -0.0145294, -0.0077290, -0.0146071, -0.0071033, -0.0057449, 0.0051255
6: 0.0035025, 0.0052286, 0.0033437, 0.0052483, -0.0013009, 0.0014581
7: 0.0059245, 0.0103903, 0.0055136, 0.0104412, -0.0033659, 0.0037726
8: 0.0035515, 0.0059000, 0.0033354, 0.0059268, -0.0017701, 0.0019840
9: -0.0087052, -0.0059820, -0.0087363, -0.0057314, -0.0023005, 0.0020525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016321, upper bound: 0.0015150
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016321, upper bound: 0.0015508
time: 1.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0016536, 1.0057963, 1.0011820, 1.0056243, -0.0030386, 0.0036817
1: -0.0008519, 0.0001803, -0.0009694, 0.0001374, -0.0007571, 0.0009174
2: -0.0110097, -0.0055392, -0.0107824, -0.0049166, -0.0048616, 0.0040124
3: 0.0012481, 0.0037380, 0.0009647, 0.0036346, -0.0018263, 0.0022128
4: -0.0016030, -0.0005442, -0.0015590, -0.0004237, -0.0009410, 0.0007766
5: -0.0148878, -0.0080074, -0.0146020, -0.0072244, -0.0061147, 0.0050465
6: 0.0035732, 0.0053195, 0.0033745, 0.0052470, -0.0012809, 0.0015520
7: 0.0061073, 0.0106256, 0.0055931, 0.0104379, -0.0033140, 0.0040154
8: 0.0036476, 0.0060237, 0.0033772, 0.0059250, -0.0017428, 0.0021117
9: -0.0088487, -0.0060935, -0.0087342, -0.0057799, -0.0024486, 0.0020208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016214, upper bound: 0.0015031
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016214, upper bound: 0.0015499
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0016177, 1.0056852, -0.0033854, 0.0031268
1: -0.0009376, 0.0001723, -0.0008609, 0.0001526, -0.0008435, 0.0007791
2: -0.0109673, -0.0050852, -0.0108629, -0.0054918, -0.0041289, 0.0044704
3: 0.0010414, 0.0037187, 0.0012265, 0.0036712, -0.0020347, 0.0018793
4: -0.0015948, -0.0004563, -0.0015746, -0.0005350, -0.0007991, 0.0008652
5: -0.0148345, -0.0074363, -0.0147032, -0.0079478, -0.0051931, 0.0056225
6: 0.0034283, 0.0053060, 0.0035581, 0.0052727, -0.0014271, 0.0013181
7: 0.0057323, 0.0105906, 0.0060682, 0.0105044, -0.0036922, 0.0034102
8: 0.0034504, 0.0060053, 0.0036270, 0.0059600, -0.0019417, 0.0017934
9: -0.0088273, -0.0058648, -0.0087748, -0.0060696, -0.0020795, 0.0022515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014932
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0017924, 1.0059015, -0.0036663, 0.0030354
1: -0.0009376, 0.0001723, -0.0008173, 0.0002065, -0.0009136, 0.0007563
2: -0.0109673, -0.0050852, -0.0111484, -0.0057227, -0.0040082, 0.0048414
3: 0.0010414, 0.0037187, 0.0013316, 0.0038012, -0.0022036, 0.0018243
4: -0.0015948, -0.0004563, -0.0016299, -0.0005797, -0.0007758, 0.0009370
5: -0.0148345, -0.0074363, -0.0150624, -0.0082382, -0.0050412, 0.0060892
6: 0.0034283, 0.0053060, 0.0036318, 0.0053638, -0.0015455, 0.0012795
7: 0.0057323, 0.0105906, 0.0062589, 0.0107402, -0.0039987, 0.0033105
8: 0.0034504, 0.0060053, 0.0037273, 0.0060840, -0.0021029, 0.0017410
9: -0.0088273, -0.0058648, -0.0089186, -0.0061859, -0.0020187, 0.0024384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014932
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0014859, 1.0055805, -0.0033320, 0.0033247
1: -0.0009376, 0.0001723, -0.0008937, 0.0001266, -0.0008303, 0.0008284
2: -0.0109673, -0.0050852, -0.0107247, -0.0053178, -0.0043902, 0.0043999
3: 0.0010414, 0.0037187, 0.0011473, 0.0036083, -0.0020026, 0.0019982
4: -0.0015948, -0.0004563, -0.0015479, -0.0005014, -0.0008497, 0.0008516
5: -0.0148345, -0.0074363, -0.0145294, -0.0077290, -0.0055217, 0.0055339
6: 0.0034283, 0.0053060, 0.0035025, 0.0052286, -0.0014046, 0.0014015
7: 0.0057323, 0.0105906, 0.0059245, 0.0103903, -0.0036341, 0.0036260
8: 0.0034504, 0.0060053, 0.0035515, 0.0059000, -0.0019111, 0.0019069
9: -0.0088273, -0.0058648, -0.0087052, -0.0059820, -0.0022111, 0.0022160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0015290
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016475
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0016536, 1.0057963, -0.0036114, 0.0032355
1: -0.0009376, 0.0001723, -0.0008519, 0.0001803, -0.0008999, 0.0008062
2: -0.0109673, -0.0050852, -0.0110097, -0.0055392, -0.0042724, 0.0047688
3: 0.0010414, 0.0037187, 0.0012481, 0.0037380, -0.0021706, 0.0019446
4: -0.0015948, -0.0004563, -0.0016030, -0.0005442, -0.0008269, 0.0009230
5: -0.0148345, -0.0074363, -0.0148878, -0.0080074, -0.0053736, 0.0059979
6: 0.0034283, 0.0053060, 0.0035732, 0.0053195, -0.0015223, 0.0013639
7: 0.0057323, 0.0105906, 0.0061073, 0.0106256, -0.0039387, 0.0035287
8: 0.0034504, 0.0060053, 0.0036476, 0.0060237, -0.0020713, 0.0018557
9: -0.0088273, -0.0058648, -0.0088487, -0.0060935, -0.0021518, 0.0024018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0015290
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016475
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0014869, 1.0059855, 1.0014930, 1.0055364, -0.0032104, 0.0036213
1: -0.0008935, 0.0002275, -0.0008920, 0.0001156, -0.0008000, 0.0009023
2: -0.0112595, -0.0053190, -0.0106664, -0.0053271, -0.0047818, 0.0042393
3: 0.0011479, 0.0038517, 0.0011515, 0.0035818, -0.0019296, 0.0021765
4: -0.0016514, -0.0005016, -0.0015366, -0.0005032, -0.0009255, 0.0008205
5: -0.0152021, -0.0077305, -0.0144561, -0.0077406, -0.0060143, 0.0053320
6: 0.0035029, 0.0053993, 0.0035055, 0.0052099, -0.0013533, 0.0015265
7: 0.0059255, 0.0108320, 0.0059321, 0.0103421, -0.0035014, 0.0039495
8: 0.0035520, 0.0061323, 0.0035555, 0.0058747, -0.0018414, 0.0020770
9: -0.0089745, -0.0059826, -0.0086758, -0.0059866, -0.0024084, 0.0021352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014655, upper bound: 0.0014768
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014655, upper bound: 0.0016312
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0011091, 1.0056273, 1.0014859, 1.0055805, -0.0034590, 0.0030861
1: -0.0009876, 0.0001382, -0.0008937, 0.0001266, -0.0008619, 0.0007690
2: -0.0107864, -0.0048204, -0.0107247, -0.0053178, -0.0040752, 0.0045676
3: 0.0009209, 0.0036364, 0.0011473, 0.0036083, -0.0020790, 0.0018548
4: -0.0015598, -0.0004051, -0.0015479, -0.0005014, -0.0007887, 0.0008841
5: -0.0146071, -0.0071033, -0.0145294, -0.0077290, -0.0051255, 0.0057449
6: 0.0033437, 0.0052483, 0.0035025, 0.0052286, -0.0014581, 0.0013009
7: 0.0055136, 0.0104412, 0.0059245, 0.0103903, -0.0037726, 0.0033659
8: 0.0033354, 0.0059268, 0.0035515, 0.0059000, -0.0019840, 0.0017701
9: -0.0087363, -0.0057314, -0.0087052, -0.0059820, -0.0020525, 0.0023005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015588, upper bound: 0.0016425
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015588, upper bound: 0.0016425
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0011820, 1.0056243, 1.0016536, 1.0057963, -0.0036817, 0.0030386
1: -0.0009694, 0.0001374, -0.0008519, 0.0001803, -0.0009174, 0.0007571
2: -0.0107824, -0.0049166, -0.0110097, -0.0055392, -0.0040124, 0.0048616
3: 0.0009647, 0.0036346, 0.0012481, 0.0037380, -0.0022128, 0.0018263
4: -0.0015590, -0.0004237, -0.0016030, -0.0005442, -0.0007766, 0.0009410
5: -0.0146020, -0.0072244, -0.0148878, -0.0080074, -0.0050465, 0.0061147
6: 0.0033745, 0.0052470, 0.0035732, 0.0053195, -0.0015520, 0.0012809
7: 0.0055931, 0.0104379, 0.0061073, 0.0106256, -0.0040154, 0.0033140
8: 0.0033772, 0.0059250, 0.0036476, 0.0060237, -0.0021117, 0.0017428
9: -0.0087342, -0.0057799, -0.0088487, -0.0060935, -0.0020208, 0.0024486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015587, upper bound: 0.0016445
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015587, upper bound: 0.0016445
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0013096, 1.0057642, -0.0030951, 0.0030951
1: -0.0009376, 0.0001723, -0.0009376, 0.0001723, -0.0007712, 0.0007712
2: -0.0109673, -0.0050852, -0.0109673, -0.0050852, -0.0040871, 0.0040871
3: 0.0010414, 0.0037187, 0.0010414, 0.0037187, -0.0018603, 0.0018603
4: -0.0015948, -0.0004563, -0.0015948, -0.0004563, -0.0007910, 0.0007910
5: -0.0148345, -0.0074363, -0.0148345, -0.0074363, -0.0051405, 0.0051405
6: 0.0034283, 0.0053060, 0.0034283, 0.0053060, -0.0013047, 0.0013047
7: 0.0057323, 0.0105906, 0.0057323, 0.0105906, -0.0033757, 0.0033757
8: 0.0034504, 0.0060053, 0.0034504, 0.0060053, -0.0017752, 0.0017752
9: -0.0088273, -0.0058648, -0.0088273, -0.0058648, -0.0020585, 0.0020585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014928
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0014869, 1.0059855, -0.0033767, 0.0030072
1: -0.0009376, 0.0001723, -0.0008935, 0.0002275, -0.0008414, 0.0007493
2: -0.0109673, -0.0050852, -0.0112595, -0.0053190, -0.0039710, 0.0044590
3: 0.0010414, 0.0037187, 0.0011479, 0.0038517, -0.0020295, 0.0018074
4: -0.0015948, -0.0004563, -0.0016514, -0.0005016, -0.0007686, 0.0008630
5: -0.0148345, -0.0074363, -0.0152021, -0.0077305, -0.0049945, 0.0056082
6: 0.0034283, 0.0053060, 0.0035029, 0.0053993, -0.0014234, 0.0012677
7: 0.0057323, 0.0105906, 0.0059255, 0.0108320, -0.0036828, 0.0032798
8: 0.0034504, 0.0060053, 0.0035520, 0.0061323, -0.0019368, 0.0017248
9: -0.0088273, -0.0058648, -0.0089745, -0.0059826, -0.0020000, 0.0022458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014928
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0011091, 1.0056273, -0.0030252, 0.0033679
1: -0.0009376, 0.0001723, -0.0009876, 0.0001382, -0.0007538, 0.0008392
2: -0.0109673, -0.0050852, -0.0107864, -0.0048204, -0.0044472, 0.0039947
3: 0.0010414, 0.0037187, 0.0009209, 0.0036364, -0.0018182, 0.0020242
4: -0.0015948, -0.0004563, -0.0015598, -0.0004051, -0.0008608, 0.0007732
5: -0.0148345, -0.0074363, -0.0146071, -0.0071033, -0.0055935, 0.0050243
6: 0.0034283, 0.0053060, 0.0033437, 0.0052483, -0.0012752, 0.0014197
7: 0.0057323, 0.0105906, 0.0055136, 0.0104412, -0.0032994, 0.0036731
8: 0.0034504, 0.0060053, 0.0033354, 0.0059268, -0.0017351, 0.0019317
9: -0.0088273, -0.0058648, -0.0087363, -0.0057314, -0.0022399, 0.0020120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0015286
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0016475
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0014869, 1.0059855, 1.0011820, 1.0056243, -0.0029866, 0.0035941
1: -0.0008935, 0.0002275, -0.0009694, 0.0001374, -0.0007442, 0.0008955
2: -0.0112595, -0.0053190, -0.0107824, -0.0049166, -0.0047459, 0.0039437
3: 0.0011479, 0.0038517, 0.0009647, 0.0036346, -0.0017950, 0.0021601
4: -0.0016514, -0.0005016, -0.0015590, -0.0004237, -0.0009186, 0.0007633
5: -0.0152021, -0.0077305, -0.0146020, -0.0072244, -0.0059691, 0.0049602
6: 0.0035029, 0.0053993, 0.0033745, 0.0052470, -0.0012589, 0.0015150
7: 0.0059255, 0.0108320, 0.0055931, 0.0104379, -0.0032573, 0.0039198
8: 0.0035520, 0.0061323, 0.0033772, 0.0059250, -0.0017130, 0.0020614
9: -0.0089745, -0.0059826, -0.0087342, -0.0057799, -0.0023903, 0.0019863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014655, upper bound: 0.0014719
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014655, upper bound: 0.0016312
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0011091, 1.0056273, 1.0010141, 1.0054914, -0.0030416, 0.0032177
1: -0.0009876, 0.0001382, -0.0010113, 0.0001044, -0.0007579, 0.0008018
2: -0.0107864, -0.0048204, -0.0106071, -0.0046948, -0.0042490, 0.0040163
3: 0.0009209, 0.0036364, 0.0008637, 0.0035548, -0.0018281, 0.0019339
4: -0.0015598, -0.0004051, -0.0015251, -0.0003808, -0.0008224, 0.0007774
5: -0.0146071, -0.0071033, -0.0143814, -0.0069453, -0.0053441, 0.0050515
6: 0.0033437, 0.0052483, 0.0033036, 0.0051910, -0.0012821, 0.0013564
7: 0.0055136, 0.0104412, 0.0054099, 0.0102931, -0.0033173, 0.0035094
8: 0.0033354, 0.0059268, 0.0032809, 0.0058489, -0.0017445, 0.0018456
9: -0.0087363, -0.0057314, -0.0086459, -0.0056682, -0.0021400, 0.0020228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0011091, 1.0056273, 1.0011091, 1.0056273, -0.0031598, 0.0031598
1: -0.0009876, 0.0001382, -0.0009876, 0.0001382, -0.0007873, 0.0007873
2: -0.0107864, -0.0048204, -0.0107864, -0.0048204, -0.0041724, 0.0041724
3: 0.0009209, 0.0036364, 0.0009209, 0.0036364, -0.0018991, 0.0018991
4: -0.0015598, -0.0004051, -0.0015598, -0.0004051, -0.0008076, 0.0008076
5: -0.0146071, -0.0071033, -0.0146071, -0.0071033, -0.0052478, 0.0052478
6: 0.0033437, 0.0052483, 0.0033437, 0.0052483, -0.0013320, 0.0013320
7: 0.0055136, 0.0104412, 0.0055136, 0.0104412, -0.0034462, 0.0034462
8: 0.0033354, 0.0059268, 0.0033354, 0.0059268, -0.0018123, 0.0018123
9: -0.0087363, -0.0057314, -0.0087363, -0.0057314, -0.0021015, 0.0021015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016445
time: 1.37 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.78 seconds
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015843, upper bound: 0.0015475
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015915, upper bound: 0.0015765
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015843, upper bound: 0.0015475
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015915, upper bound: 0.0015781
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015505, upper bound: 0.0014484
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0016108, upper bound: 0.0015312
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015505, upper bound: 0.0014484
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0016108, upper bound: 0.0015312
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015893, upper bound: 0.0014484
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0016423, upper bound: 0.0015312
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015893, upper bound: 0.0014484
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0016423, upper bound: 0.0015312
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015672, upper bound: 0.0014084
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015672, upper bound: 0.0015194
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0016321, upper bound: 0.0015150
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0016321, upper bound: 0.0015508
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0016214, upper bound: 0.0015031
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0016214, upper bound: 0.0015499
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014932
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014932
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0015290
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016475
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0015290
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016475
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014655, upper bound: 0.0014768
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014655, upper bound: 0.0016312
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015588, upper bound: 0.0016425
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015588, upper bound: 0.0016425
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015587, upper bound: 0.0016445
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015587, upper bound: 0.0016445
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014928
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014928
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0015286
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0016475
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014655, upper bound: 0.0014719
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0014655, upper bound: 0.0016312
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016445

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0014908, 1.0055364, 1.0013096, 1.0057642, -0.0033199, 0.0032554
1: -0.0008925, 0.0001156, -0.0009376, 0.0001723, -0.0008272, 0.0008112
2: -0.0106665, -0.0053243, -0.0109673, -0.0050852, -0.0042987, 0.0043839
3: 0.0011503, 0.0035818, 0.0010414, 0.0037187, -0.0019954, 0.0019566
4: -0.0015366, -0.0005026, -0.0015948, -0.0004563, -0.0008320, 0.0008485
5: -0.0144563, -0.0077371, -0.0148345, -0.0074363, -0.0054066, 0.0055138
6: 0.0035046, 0.0052100, 0.0034283, 0.0053060, -0.0013995, 0.0013723
7: 0.0059298, 0.0103422, 0.0057323, 0.0105906, -0.0036209, 0.0035504
8: 0.0035543, 0.0058747, 0.0034504, 0.0060053, -0.0019042, 0.0018671
9: -0.0086759, -0.0059852, -0.0088273, -0.0058648, -0.0021650, 0.0022080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017013, upper bound: 0.0016038
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017013, upper bound: 0.0016061
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0014908, 1.0055364, 1.0014869, 1.0059855, -0.0035960, 0.0031421
1: -0.0008925, 0.0001156, -0.0008935, 0.0002275, -0.0008960, 0.0007829
2: -0.0106665, -0.0053243, -0.0112595, -0.0053190, -0.0041491, 0.0047485
3: 0.0011503, 0.0035818, 0.0011479, 0.0038517, -0.0021613, 0.0018885
4: -0.0015366, -0.0005026, -0.0016514, -0.0005016, -0.0008031, 0.0009191
5: -0.0144563, -0.0077371, -0.0152021, -0.0077305, -0.0052185, 0.0059724
6: 0.0035046, 0.0052100, 0.0035029, 0.0053993, -0.0015158, 0.0013245
7: 0.0059298, 0.0103422, 0.0059255, 0.0108320, -0.0039220, 0.0034269
8: 0.0035543, 0.0058747, 0.0035520, 0.0061323, -0.0020625, 0.0018022
9: -0.0086759, -0.0059852, -0.0089745, -0.0059826, -0.0020897, 0.0023916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015141, upper bound: 0.0014961
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015141, upper bound: 0.0015312
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0013630, 1.0054271, 1.0011091, 1.0056273, -0.0031947, 0.0033153
1: -0.0009243, 0.0000883, -0.0009876, 0.0001382, -0.0007960, 0.0008261
2: -0.0105222, -0.0051555, -0.0107864, -0.0048204, -0.0043779, 0.0042186
3: 0.0010735, 0.0035161, 0.0009209, 0.0036364, -0.0019201, 0.0019926
4: -0.0015087, -0.0004700, -0.0015598, -0.0004051, -0.0008473, 0.0008165
5: -0.0142747, -0.0075248, -0.0146071, -0.0071033, -0.0055062, 0.0053059
6: 0.0034507, 0.0051639, 0.0033437, 0.0052483, -0.0013467, 0.0013975
7: 0.0057904, 0.0102229, 0.0055136, 0.0104412, -0.0034843, 0.0036158
8: 0.0034810, 0.0058120, 0.0033354, 0.0059268, -0.0018324, 0.0019015
9: -0.0086032, -0.0059002, -0.0087363, -0.0057314, -0.0022049, 0.0021247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015149
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015150
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0014908, 1.0055364, 1.0011091, 1.0056273, -0.0030813, 0.0033897
1: -0.0008925, 0.0001156, -0.0009876, 0.0001382, -0.0007678, 0.0008446
2: -0.0106665, -0.0053243, -0.0107864, -0.0048204, -0.0044760, 0.0040688
3: 0.0011503, 0.0035818, 0.0009209, 0.0036364, -0.0018519, 0.0020373
4: -0.0015366, -0.0005026, -0.0015598, -0.0004051, -0.0008663, 0.0007875
5: -0.0144563, -0.0077371, -0.0146071, -0.0071033, -0.0056297, 0.0051175
6: 0.0035046, 0.0052100, 0.0033437, 0.0052483, -0.0012989, 0.0014289
7: 0.0059298, 0.0103422, 0.0055136, 0.0104412, -0.0033606, 0.0036969
8: 0.0035543, 0.0058747, 0.0033354, 0.0059268, -0.0017673, 0.0019442
9: -0.0086759, -0.0059852, -0.0087363, -0.0057314, -0.0022544, 0.0020493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015504
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015508
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0015507, 1.0056390, 1.0011820, 1.0056243, -0.0031443, 0.0035365
1: -0.0008776, 0.0001411, -0.0009694, 0.0001374, -0.0007835, 0.0008812
2: -0.0108018, -0.0054034, -0.0107824, -0.0049166, -0.0046699, 0.0041520
3: 0.0011863, 0.0036434, 0.0009647, 0.0036346, -0.0018898, 0.0021255
4: -0.0015628, -0.0005179, -0.0015590, -0.0004237, -0.0009039, 0.0008036
5: -0.0146264, -0.0078366, -0.0146020, -0.0072244, -0.0058735, 0.0052222
6: 0.0035299, 0.0052532, 0.0033745, 0.0052470, -0.0013254, 0.0014908
7: 0.0059952, 0.0104539, 0.0055931, 0.0104379, -0.0034293, 0.0038571
8: 0.0035887, 0.0059335, 0.0033772, 0.0059250, -0.0018034, 0.0020284
9: -0.0087440, -0.0060251, -0.0087342, -0.0057799, -0.0023520, 0.0020912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015031
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015031
time: 1.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0016576, 1.0057553, 1.0011820, 1.0056243, -0.0030347, 0.0036098
1: -0.0008509, 0.0001701, -0.0009694, 0.0001374, -0.0007562, 0.0008995
2: -0.0109555, -0.0055445, -0.0107824, -0.0049166, -0.0047667, 0.0040072
3: 0.0012505, 0.0037134, 0.0009647, 0.0036346, -0.0018239, 0.0021696
4: -0.0015925, -0.0005452, -0.0015590, -0.0004237, -0.0009226, 0.0007756
5: -0.0148197, -0.0080141, -0.0146020, -0.0072244, -0.0059952, 0.0050401
6: 0.0035749, 0.0053022, 0.0033745, 0.0052470, -0.0012792, 0.0015217
7: 0.0061117, 0.0105809, 0.0055931, 0.0104379, -0.0033097, 0.0039370
8: 0.0036499, 0.0060002, 0.0033772, 0.0059250, -0.0017406, 0.0020704
9: -0.0088214, -0.0060961, -0.0087342, -0.0057799, -0.0024007, 0.0020183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015461
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015461
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0016177, 1.0056852, -0.0033814, 0.0030798
1: -0.0009370, 0.0001626, -0.0008609, 0.0001526, -0.0008426, 0.0007674
2: -0.0109158, -0.0050884, -0.0108629, -0.0054918, -0.0040669, 0.0044651
3: 0.0010429, 0.0036953, 0.0012265, 0.0036712, -0.0020323, 0.0018511
4: -0.0015848, -0.0004570, -0.0015746, -0.0005350, -0.0007871, 0.0008642
5: -0.0147698, -0.0074404, -0.0147032, -0.0079478, -0.0051150, 0.0056159
6: 0.0034293, 0.0052896, 0.0035581, 0.0052727, -0.0014254, 0.0012983
7: 0.0057350, 0.0105481, 0.0060682, 0.0105044, -0.0036879, 0.0033590
8: 0.0034518, 0.0059830, 0.0036270, 0.0059600, -0.0019394, 0.0017665
9: -0.0088014, -0.0058664, -0.0087748, -0.0060696, -0.0020483, 0.0022489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017096
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017128
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0017924, 1.0059015, -0.0036623, 0.0029869
1: -0.0009370, 0.0001626, -0.0008173, 0.0002065, -0.0009126, 0.0007443
2: -0.0109158, -0.0050884, -0.0111484, -0.0057227, -0.0039442, 0.0048361
3: 0.0010429, 0.0036953, 0.0013316, 0.0038012, -0.0022012, 0.0017952
4: -0.0015848, -0.0004570, -0.0016299, -0.0005797, -0.0007634, 0.0009360
5: -0.0147698, -0.0074404, -0.0150624, -0.0082382, -0.0049607, 0.0060825
6: 0.0034293, 0.0052896, 0.0036318, 0.0053638, -0.0015438, 0.0012591
7: 0.0057350, 0.0105481, 0.0062589, 0.0107402, -0.0039943, 0.0032576
8: 0.0034518, 0.0059830, 0.0037273, 0.0060840, -0.0021006, 0.0017132
9: -0.0088014, -0.0058664, -0.0089186, -0.0061859, -0.0019865, 0.0024357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015723
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0016203
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0014859, 1.0055805, -0.0033280, 0.0032659
1: -0.0009370, 0.0001626, -0.0008937, 0.0001266, -0.0008293, 0.0008138
2: -0.0109158, -0.0050884, -0.0107247, -0.0053178, -0.0043125, 0.0043946
3: 0.0010429, 0.0036953, 0.0011473, 0.0036083, -0.0020003, 0.0019629
4: -0.0015848, -0.0004570, -0.0015479, -0.0005014, -0.0008347, 0.0008506
5: -0.0147698, -0.0074404, -0.0145294, -0.0077290, -0.0054240, 0.0055273
6: 0.0034293, 0.0052896, 0.0035025, 0.0052286, -0.0014029, 0.0013767
7: 0.0057350, 0.0105481, 0.0059245, 0.0103903, -0.0036297, 0.0035619
8: 0.0034518, 0.0059830, 0.0035515, 0.0059000, -0.0019088, 0.0018732
9: -0.0088014, -0.0058664, -0.0087052, -0.0059820, -0.0021720, 0.0022134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017308
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017330
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0016536, 1.0057963, -0.0036074, 0.0031738
1: -0.0009370, 0.0001626, -0.0008519, 0.0001803, -0.0008989, 0.0007908
2: -0.0109158, -0.0050884, -0.0110097, -0.0055392, -0.0041909, 0.0047635
3: 0.0010429, 0.0036953, 0.0012481, 0.0037380, -0.0021682, 0.0019075
4: -0.0015848, -0.0004570, -0.0016030, -0.0005442, -0.0008111, 0.0009220
5: -0.0147698, -0.0074404, -0.0148878, -0.0080074, -0.0052711, 0.0059913
6: 0.0034293, 0.0052896, 0.0035732, 0.0053195, -0.0015207, 0.0013379
7: 0.0057350, 0.0105481, 0.0061073, 0.0106256, -0.0039344, 0.0034615
8: 0.0034518, 0.0059830, 0.0036476, 0.0060237, -0.0020691, 0.0018203
9: -0.0088014, -0.0058664, -0.0088487, -0.0060935, -0.0021108, 0.0023992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014485, upper bound: 0.0016108
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014485, upper bound: 0.0016475
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0014889, 1.0059466, 1.0014930, 1.0055364, -0.0032043, 0.0035569
1: -0.0008930, 0.0002178, -0.0008920, 0.0001156, -0.0007984, 0.0008863
2: -0.0112082, -0.0053218, -0.0106664, -0.0053271, -0.0046968, 0.0042313
3: 0.0011491, 0.0038284, 0.0011515, 0.0035818, -0.0019259, 0.0021378
4: -0.0016414, -0.0005021, -0.0015366, -0.0005032, -0.0009091, 0.0008190
5: -0.0151375, -0.0077339, -0.0144561, -0.0077406, -0.0059073, 0.0053218
6: 0.0035038, 0.0053829, 0.0035055, 0.0052099, -0.0013507, 0.0014993
7: 0.0059277, 0.0107895, 0.0059321, 0.0103421, -0.0034948, 0.0038793
8: 0.0035532, 0.0061100, 0.0035555, 0.0058747, -0.0018379, 0.0020401
9: -0.0089487, -0.0059840, -0.0086758, -0.0059866, -0.0023656, 0.0021311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
time: 1.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0014859, 1.0055805, -0.0033776, 0.0030842
1: -0.0009695, 0.0001375, -0.0008937, 0.0001266, -0.0008416, 0.0007685
2: -0.0107827, -0.0049161, -0.0107247, -0.0053178, -0.0040726, 0.0044601
3: 0.0009645, 0.0036347, 0.0011473, 0.0036083, -0.0020300, 0.0018537
4: -0.0015591, -0.0004236, -0.0015479, -0.0005014, -0.0007883, 0.0008632
5: -0.0146023, -0.0072237, -0.0145294, -0.0077290, -0.0051223, 0.0056096
6: 0.0033743, 0.0052471, 0.0035025, 0.0052286, -0.0014238, 0.0013001
7: 0.0055927, 0.0104381, 0.0059245, 0.0103903, -0.0036838, 0.0033637
8: 0.0033770, 0.0059252, 0.0035515, 0.0059000, -0.0019373, 0.0017690
9: -0.0087344, -0.0057796, -0.0087052, -0.0059820, -0.0020512, 0.0022463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015298, upper bound: 0.0016166
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015298, upper bound: 0.0016471
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0014859, 1.0055805, -0.0032701, 0.0033613
1: -0.0009278, 0.0001933, -0.0008937, 0.0001266, -0.0008148, 0.0008375
2: -0.0110784, -0.0051370, -0.0107247, -0.0053178, -0.0044385, 0.0043181
3: 0.0010650, 0.0037693, 0.0011473, 0.0036083, -0.0019654, 0.0020202
4: -0.0016163, -0.0004664, -0.0015479, -0.0005014, -0.0008591, 0.0008358
5: -0.0149742, -0.0075016, -0.0145294, -0.0077290, -0.0055825, 0.0054310
6: 0.0034448, 0.0053415, 0.0035025, 0.0052286, -0.0013784, 0.0014169
7: 0.0057752, 0.0106823, 0.0059245, 0.0103903, -0.0035665, 0.0036660
8: 0.0034729, 0.0060536, 0.0035515, 0.0059000, -0.0018756, 0.0019279
9: -0.0088833, -0.0058909, -0.0087052, -0.0059820, -0.0022355, 0.0021748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015298, upper bound: 0.0016166
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015298, upper bound: 0.0016471
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0016536, 1.0057963, -0.0036615, 0.0029914
1: -0.0009695, 0.0001375, -0.0008519, 0.0001803, -0.0009124, 0.0007454
2: -0.0107827, -0.0049161, -0.0110097, -0.0055392, -0.0039501, 0.0048350
3: 0.0009645, 0.0036347, 0.0012481, 0.0037380, -0.0022007, 0.0017979
4: -0.0015591, -0.0004236, -0.0016030, -0.0005442, -0.0007645, 0.0009358
5: -0.0146023, -0.0072237, -0.0148878, -0.0080074, -0.0049682, 0.0060812
6: 0.0033743, 0.0052471, 0.0035732, 0.0053195, -0.0015435, 0.0012610
7: 0.0055927, 0.0104381, 0.0061073, 0.0106256, -0.0039934, 0.0032625
8: 0.0033770, 0.0059252, 0.0036476, 0.0060237, -0.0021001, 0.0017157
9: -0.0087344, -0.0057796, -0.0088487, -0.0060935, -0.0019895, 0.0024352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016445
time: 1.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0016536, 1.0057963, -0.0034052, 0.0031189
1: -0.0009278, 0.0001933, -0.0008519, 0.0001803, -0.0008485, 0.0007771
2: -0.0110784, -0.0051370, -0.0110097, -0.0055392, -0.0041185, 0.0044966
3: 0.0010650, 0.0037693, 0.0012481, 0.0037380, -0.0020466, 0.0018745
4: -0.0016163, -0.0004664, -0.0016030, -0.0005442, -0.0007971, 0.0008703
5: -0.0149742, -0.0075016, -0.0148878, -0.0080074, -0.0051799, 0.0056555
6: 0.0034448, 0.0053415, 0.0035732, 0.0053195, -0.0014354, 0.0013147
7: 0.0057752, 0.0106823, 0.0061073, 0.0106256, -0.0037139, 0.0034016
8: 0.0034729, 0.0060536, 0.0036476, 0.0060237, -0.0019531, 0.0017889
9: -0.0088833, -0.0058909, -0.0088487, -0.0060935, -0.0020743, 0.0022647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016058
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016425
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0013096, 1.0057642, -0.0030911, 0.0030866
1: -0.0009370, 0.0001626, -0.0009376, 0.0001723, -0.0007702, 0.0007691
2: -0.0109158, -0.0050884, -0.0109673, -0.0050852, -0.0040758, 0.0040818
3: 0.0010429, 0.0036953, 0.0010414, 0.0037187, -0.0018579, 0.0018551
4: -0.0015848, -0.0004570, -0.0015948, -0.0004563, -0.0007889, 0.0007900
5: -0.0147698, -0.0074404, -0.0148345, -0.0074363, -0.0051263, 0.0051339
6: 0.0034293, 0.0052896, 0.0034283, 0.0053060, -0.0013030, 0.0013011
7: 0.0057350, 0.0105481, 0.0057323, 0.0105906, -0.0033713, 0.0033663
8: 0.0034518, 0.0059830, 0.0034504, 0.0060053, -0.0017730, 0.0017703
9: -0.0088014, -0.0058664, -0.0088273, -0.0058648, -0.0020528, 0.0020558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017096
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017128
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0014869, 1.0059855, -0.0033728, 0.0029954
1: -0.0009370, 0.0001626, -0.0008935, 0.0002275, -0.0008404, 0.0007464
2: -0.0109158, -0.0050884, -0.0112595, -0.0053190, -0.0039554, 0.0044537
3: 0.0010429, 0.0036953, 0.0011479, 0.0038517, -0.0020271, 0.0018003
4: -0.0015848, -0.0004570, -0.0016514, -0.0005016, -0.0007656, 0.0008620
5: -0.0147698, -0.0074404, -0.0152021, -0.0077305, -0.0049749, 0.0056016
6: 0.0034293, 0.0052896, 0.0035029, 0.0053993, -0.0014217, 0.0012627
7: 0.0057350, 0.0105481, 0.0059255, 0.0108320, -0.0036785, 0.0032670
8: 0.0034518, 0.0059830, 0.0035520, 0.0061323, -0.0019345, 0.0017181
9: -0.0088014, -0.0058664, -0.0089745, -0.0059826, -0.0019922, 0.0022431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0015723
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0016203
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0011091, 1.0056273, -0.0030212, 0.0033364
1: -0.0009370, 0.0001626, -0.0009876, 0.0001382, -0.0007528, 0.0008313
2: -0.0109158, -0.0050884, -0.0107864, -0.0048204, -0.0044056, 0.0039894
3: 0.0010429, 0.0036953, 0.0009209, 0.0036364, -0.0018158, 0.0020053
4: -0.0015848, -0.0004570, -0.0015598, -0.0004051, -0.0008527, 0.0007721
5: -0.0147698, -0.0074404, -0.0146071, -0.0071033, -0.0055411, 0.0050176
6: 0.0034293, 0.0052896, 0.0033437, 0.0052483, -0.0012735, 0.0014064
7: 0.0057350, 0.0105481, 0.0055136, 0.0104412, -0.0032950, 0.0036388
8: 0.0034518, 0.0059830, 0.0033354, 0.0059268, -0.0017328, 0.0019136
9: -0.0088014, -0.0058664, -0.0087363, -0.0057314, -0.0022189, 0.0020093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014465, upper bound: 0.0016475
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014465, upper bound: 0.0016475
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0014889, 1.0059466, 1.0011820, 1.0056243, -0.0029810, 0.0035611
1: -0.0008930, 0.0002178, -0.0009694, 0.0001374, -0.0007428, 0.0008873
2: -0.0112082, -0.0053218, -0.0107824, -0.0049166, -0.0047023, 0.0039364
3: 0.0011491, 0.0038284, 0.0009647, 0.0036346, -0.0017917, 0.0021403
4: -0.0016414, -0.0005021, -0.0015590, -0.0004237, -0.0009101, 0.0007619
5: -0.0151375, -0.0077339, -0.0146020, -0.0072244, -0.0059143, 0.0049510
6: 0.0035038, 0.0053829, 0.0033745, 0.0052470, -0.0012566, 0.0015011
7: 0.0059277, 0.0107895, 0.0055931, 0.0104379, -0.0032512, 0.0038838
8: 0.0035532, 0.0061100, 0.0033772, 0.0059250, -0.0017098, 0.0020425
9: -0.0089487, -0.0059840, -0.0087342, -0.0057799, -0.0023683, 0.0019826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0010141, 1.0054914, -0.0029607, 0.0032155
1: -0.0009695, 0.0001375, -0.0010113, 0.0001044, -0.0007377, 0.0008012
2: -0.0107827, -0.0049161, -0.0106071, -0.0046948, -0.0042461, 0.0039095
3: 0.0009645, 0.0036347, 0.0008637, 0.0035548, -0.0017795, 0.0019326
4: -0.0015591, -0.0004236, -0.0015251, -0.0003808, -0.0008218, 0.0007567
5: -0.0146023, -0.0072237, -0.0143814, -0.0069453, -0.0053404, 0.0049172
6: 0.0033743, 0.0052471, 0.0033036, 0.0051910, -0.0012480, 0.0013555
7: 0.0055927, 0.0104381, 0.0054099, 0.0102931, -0.0032290, 0.0035070
8: 0.0033770, 0.0059252, 0.0032809, 0.0058489, -0.0016981, 0.0018443
9: -0.0087344, -0.0057796, -0.0086459, -0.0056682, -0.0021385, 0.0019691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016058
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0011091, 1.0056273, -0.0030789, 0.0031578
1: -0.0009695, 0.0001375, -0.0009876, 0.0001382, -0.0007672, 0.0007868
2: -0.0107827, -0.0049161, -0.0107864, -0.0048204, -0.0041699, 0.0040657
3: 0.0009645, 0.0036347, 0.0009209, 0.0036364, -0.0018505, 0.0018979
4: -0.0015591, -0.0004236, -0.0015598, -0.0004051, -0.0008071, 0.0007869
5: -0.0146023, -0.0072237, -0.0146071, -0.0071033, -0.0052446, 0.0051136
6: 0.0033743, 0.0052471, 0.0033437, 0.0052483, -0.0012979, 0.0013311
7: 0.0055927, 0.0104381, 0.0055136, 0.0104412, -0.0033580, 0.0034441
8: 0.0033770, 0.0059252, 0.0033354, 0.0059268, -0.0017660, 0.0018112
9: -0.0087344, -0.0057796, -0.0087363, -0.0057314, -0.0021002, 0.0020477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015526, upper bound: 0.0016425
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015526, upper bound: 0.0016445
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0011820, 1.0056243, -0.0030372, 0.0033836
1: -0.0009278, 0.0001933, -0.0009694, 0.0001374, -0.0007568, 0.0008431
2: -0.0110784, -0.0051370, -0.0107824, -0.0049166, -0.0044680, 0.0040105
3: 0.0010650, 0.0037693, 0.0009647, 0.0036346, -0.0018254, 0.0020336
4: -0.0016163, -0.0004664, -0.0015590, -0.0004237, -0.0008648, 0.0007762
5: -0.0149742, -0.0075016, -0.0146020, -0.0072244, -0.0056195, 0.0050442
6: 0.0034448, 0.0053415, 0.0033745, 0.0052470, -0.0012803, 0.0014263
7: 0.0057752, 0.0106823, 0.0055931, 0.0104379, -0.0033125, 0.0036902
8: 0.0034729, 0.0060536, 0.0033772, 0.0059250, -0.0017420, 0.0019407
9: -0.0088833, -0.0058909, -0.0087342, -0.0057799, -0.0022503, 0.0020199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015560, upper bound: 0.0016425
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015560, upper bound: 0.0016445
time: 1.13 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.54 seconds
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0017013, upper bound: 0.0016038
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0017013, upper bound: 0.0016061
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015141, upper bound: 0.0014961
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015141, upper bound: 0.0015312
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015149
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015150
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015504
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015845, upper bound: 0.0015508
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015031
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015031
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015461
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015461
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017096
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017128
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015723
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0016203
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017308
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017330
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014485, upper bound: 0.0016108
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014485, upper bound: 0.0016475
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015298, upper bound: 0.0016166
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015298, upper bound: 0.0016471
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015298, upper bound: 0.0016166
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015298, upper bound: 0.0016471
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016445
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016058
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016425
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017096
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015916, upper bound: 0.0017128
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0015723
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0016203
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014465, upper bound: 0.0016475
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014465, upper bound: 0.0016475
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016058
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015526, upper bound: 0.0016425
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015526, upper bound: 0.0016445
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015560, upper bound: 0.0016425
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.54
Output dim: 0, lower bound: -0.0015560, upper bound: 0.0016445

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0014908, 1.0055364, 1.0012008, 1.0055921, -0.0031527, 0.0033461
1: -0.0008925, 0.0001156, -0.0009647, 0.0001294, -0.0007856, 0.0008337
2: -0.0106665, -0.0053243, -0.0107400, -0.0049413, -0.0044184, 0.0041630
3: 0.0011503, 0.0035818, 0.0009760, 0.0036152, -0.0018948, 0.0020111
4: -0.0015366, -0.0005026, -0.0015508, -0.0004285, -0.0008552, 0.0008057
5: -0.0144563, -0.0077371, -0.0145486, -0.0072555, -0.0055572, 0.0052360
6: 0.0035046, 0.0052100, 0.0033823, 0.0052334, -0.0013290, 0.0014105
7: 0.0059298, 0.0103422, 0.0056135, 0.0104028, -0.0034384, 0.0036494
8: 0.0035543, 0.0058747, 0.0033880, 0.0059066, -0.0018082, 0.0019192
9: -0.0086759, -0.0059852, -0.0087129, -0.0057924, -0.0022254, 0.0020967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014786, upper bound: 0.0015052
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014670, upper bound: 0.0014429
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0014908, 1.0055364, 1.0013121, 1.0057251, -0.0032603, 0.0032514
1: -0.0008925, 0.0001156, -0.0009370, 0.0001626, -0.0008124, 0.0008102
2: -0.0106665, -0.0053243, -0.0109158, -0.0050884, -0.0042935, 0.0043052
3: 0.0011503, 0.0035818, 0.0010429, 0.0036953, -0.0019595, 0.0019542
4: -0.0015366, -0.0005026, -0.0015848, -0.0004570, -0.0008310, 0.0008333
5: -0.0144563, -0.0077371, -0.0147698, -0.0074404, -0.0054001, 0.0054148
6: 0.0035046, 0.0052100, 0.0034293, 0.0052896, -0.0013743, 0.0013706
7: 0.0059298, 0.0103422, 0.0057350, 0.0105481, -0.0035558, 0.0035461
8: 0.0035543, 0.0058747, 0.0034518, 0.0059830, -0.0018700, 0.0018649
9: -0.0086759, -0.0059852, -0.0088014, -0.0058664, -0.0021624, 0.0021683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014786, upper bound: 0.0015394
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014670, upper bound: 0.0015165
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0014846, 1.0055283, -0.0032284, 0.0032083
1: -0.0009370, 0.0001626, -0.0008940, 0.0001135, -0.0008044, 0.0007994
2: -0.0109158, -0.0050884, -0.0106557, -0.0053162, -0.0042365, 0.0042631
3: 0.0010429, 0.0036953, 0.0011466, 0.0035769, -0.0019404, 0.0019283
4: -0.0015848, -0.0004570, -0.0015345, -0.0005011, -0.0008200, 0.0008251
5: -0.0147698, -0.0074404, -0.0144427, -0.0077270, -0.0053284, 0.0053618
6: 0.0034293, 0.0052896, 0.0035020, 0.0052065, -0.0013609, 0.0013524
7: 0.0057350, 0.0105481, 0.0059232, 0.0103333, -0.0035210, 0.0034991
8: 0.0034518, 0.0059830, 0.0035508, 0.0058700, -0.0018517, 0.0018401
9: -0.0088014, -0.0058664, -0.0086704, -0.0059812, -0.0021337, 0.0021471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0015835
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0015304
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0016202, 1.0056431, -0.0033142, 0.0030754
1: -0.0009370, 0.0001626, -0.0008602, 0.0001422, -0.0008258, 0.0007663
2: -0.0109158, -0.0050884, -0.0108075, -0.0054952, -0.0040610, 0.0043764
3: 0.0010429, 0.0036953, 0.0012280, 0.0036460, -0.0019919, 0.0018484
4: -0.0015848, -0.0004570, -0.0015639, -0.0005357, -0.0007860, 0.0008470
5: -0.0147698, -0.0074404, -0.0146335, -0.0079520, -0.0051076, 0.0055043
6: 0.0034293, 0.0052896, 0.0035591, 0.0052550, -0.0013971, 0.0012964
7: 0.0057350, 0.0105481, 0.0060709, 0.0104586, -0.0036146, 0.0033541
8: 0.0034518, 0.0059830, 0.0036285, 0.0059359, -0.0019009, 0.0017639
9: -0.0088014, -0.0058664, -0.0087468, -0.0060713, -0.0020453, 0.0022042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0016376
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0016230
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0017946, 1.0058603, -0.0035939, 0.0029812
1: -0.0009370, 0.0001626, -0.0008168, 0.0001963, -0.0008955, 0.0007428
2: -0.0109158, -0.0050884, -0.0110943, -0.0057254, -0.0039367, 0.0047457
3: 0.0010429, 0.0036953, 0.0013328, 0.0037765, -0.0021600, 0.0017918
4: -0.0015848, -0.0004570, -0.0016194, -0.0005803, -0.0007619, 0.0009185
5: -0.0147698, -0.0074404, -0.0149942, -0.0082416, -0.0049513, 0.0059688
6: 0.0034293, 0.0052896, 0.0036326, 0.0053465, -0.0015149, 0.0012567
7: 0.0057350, 0.0105481, 0.0062611, 0.0106955, -0.0039196, 0.0032514
8: 0.0034518, 0.0059830, 0.0037285, 0.0060605, -0.0020613, 0.0017099
9: -0.0088014, -0.0058664, -0.0088913, -0.0061872, -0.0019827, 0.0023902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 9.99 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009886, upper bound: 0.0009878
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008009, upper bound: 0.0009020
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0013630, 1.0054271, -0.0031748, 0.0033687
1: -0.0009370, 0.0001626, -0.0009243, 0.0000883, -0.0007911, 0.0008394
2: -0.0109158, -0.0050884, -0.0105222, -0.0051555, -0.0044484, 0.0041923
3: 0.0010429, 0.0036953, 0.0010735, 0.0035161, -0.0019082, 0.0020247
4: -0.0015848, -0.0004570, -0.0015087, -0.0004700, -0.0008610, 0.0008114
5: -0.0147698, -0.0074404, -0.0142747, -0.0075248, -0.0055949, 0.0052728
6: 0.0034293, 0.0052896, 0.0034507, 0.0051639, -0.0013383, 0.0014200
7: 0.0057350, 0.0105481, 0.0057904, 0.0102229, -0.0034626, 0.0036741
8: 0.0034518, 0.0059830, 0.0034810, 0.0058120, -0.0018209, 0.0019322
9: -0.0088014, -0.0058664, -0.0086032, -0.0059002, -0.0022404, 0.0021115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0016163
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0015725
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0014908, 1.0055364, -0.0032514, 0.0032603
1: -0.0009370, 0.0001626, -0.0008925, 0.0001156, -0.0008102, 0.0008124
2: -0.0109158, -0.0050884, -0.0106665, -0.0053243, -0.0043052, 0.0042935
3: 0.0010429, 0.0036953, 0.0011503, 0.0035818, -0.0019542, 0.0019595
4: -0.0015848, -0.0004570, -0.0015366, -0.0005026, -0.0008333, 0.0008310
5: -0.0147698, -0.0074404, -0.0144563, -0.0077371, -0.0054148, 0.0054001
6: 0.0034293, 0.0052896, 0.0035046, 0.0052100, -0.0013706, 0.0013743
7: 0.0057350, 0.0105481, 0.0059298, 0.0103422, -0.0035461, 0.0035558
8: 0.0034518, 0.0059830, 0.0035543, 0.0058747, -0.0018649, 0.0018700
9: -0.0088014, -0.0058664, -0.0086759, -0.0059852, -0.0021683, 0.0021624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0016601
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0016494
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0016576, 1.0057553, -0.0035320, 0.0031694
1: -0.0009370, 0.0001626, -0.0008509, 0.0001701, -0.0008801, 0.0007897
2: -0.0109158, -0.0050884, -0.0109555, -0.0055445, -0.0041852, 0.0046640
3: 0.0010429, 0.0036953, 0.0012505, 0.0037134, -0.0021229, 0.0019049
4: -0.0015848, -0.0004570, -0.0015925, -0.0005452, -0.0008100, 0.0009027
5: -0.0147698, -0.0074404, -0.0148197, -0.0080141, -0.0052639, 0.0058661
6: 0.0034293, 0.0052896, 0.0035749, 0.0053022, -0.0014889, 0.0013360
7: 0.0057350, 0.0105481, 0.0061117, 0.0105809, -0.0038522, 0.0034567
8: 0.0034518, 0.0059830, 0.0036499, 0.0060002, -0.0020258, 0.0018179
9: -0.0088014, -0.0058664, -0.0088214, -0.0060961, -0.0021079, 0.0023491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 10.19 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009903, upper bound: 0.0010194
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008122, upper bound: 0.0009449
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0014889, 1.0059466, 1.0014908, 1.0055364, -0.0031360, 0.0035351
1: -0.0008930, 0.0002178, -0.0008925, 0.0001156, -0.0007814, 0.0008809
2: -0.0112082, -0.0053218, -0.0106665, -0.0053243, -0.0046681, 0.0041411
3: 0.0011491, 0.0038284, 0.0011503, 0.0035818, -0.0018848, 0.0021247
4: -0.0016414, -0.0005021, -0.0015366, -0.0005026, -0.0009035, 0.0008015
5: -0.0151375, -0.0077339, -0.0144563, -0.0077371, -0.0058713, 0.0052084
6: 0.0035038, 0.0053829, 0.0035046, 0.0052100, -0.0013219, 0.0014902
7: 0.0059277, 0.0107895, 0.0059298, 0.0103422, -0.0034203, 0.0038556
8: 0.0035532, 0.0061100, 0.0035543, 0.0058747, -0.0017987, 0.0020276
9: -0.0089487, -0.0059840, -0.0086759, -0.0059852, -0.0023511, 0.0020857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0014889, 1.0059466, 1.0016576, 1.0057553, -0.0032644, 0.0032986
1: -0.0008930, 0.0002178, -0.0008509, 0.0001701, -0.0008134, 0.0008219
2: -0.0112082, -0.0053218, -0.0109555, -0.0055445, -0.0043557, 0.0043106
3: 0.0011491, 0.0038284, 0.0012505, 0.0037134, -0.0019620, 0.0019825
4: -0.0016414, -0.0005021, -0.0015925, -0.0005452, -0.0008430, 0.0008343
5: -0.0151375, -0.0077339, -0.0148197, -0.0080141, -0.0054783, 0.0054217
6: 0.0035038, 0.0053829, 0.0035749, 0.0053022, -0.0013761, 0.0013905
7: 0.0059277, 0.0107895, 0.0061117, 0.0105809, -0.0035603, 0.0035975
8: 0.0035532, 0.0061100, 0.0036499, 0.0060002, -0.0018723, 0.0018919
9: -0.0089487, -0.0059840, -0.0088214, -0.0060961, -0.0021938, 0.0021711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0014908, 1.0055364, -0.0033097, 0.0030793
1: -0.0009695, 0.0001375, -0.0008925, 0.0001156, -0.0008247, 0.0007673
2: -0.0107827, -0.0049161, -0.0106665, -0.0053243, -0.0040662, 0.0043705
3: 0.0009645, 0.0036347, 0.0011503, 0.0035818, -0.0019892, 0.0018508
4: -0.0015591, -0.0004236, -0.0015366, -0.0005026, -0.0007870, 0.0008459
5: -0.0146023, -0.0072237, -0.0144563, -0.0077371, -0.0051142, 0.0054969
6: 0.0033743, 0.0052471, 0.0035046, 0.0052100, -0.0013952, 0.0012981
7: 0.0055927, 0.0104381, 0.0059298, 0.0103422, -0.0036097, 0.0033585
8: 0.0033770, 0.0059252, 0.0035543, 0.0058747, -0.0018983, 0.0017662
9: -0.0087344, -0.0057796, -0.0086759, -0.0059852, -0.0020480, 0.0022012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0016483
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014496, upper bound: 0.0016395
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0014908, 1.0055364, -0.0032017, 0.0033564
1: -0.0009278, 0.0001933, -0.0008925, 0.0001156, -0.0007978, 0.0008363
2: -0.0110784, -0.0051370, -0.0106665, -0.0053243, -0.0044321, 0.0042278
3: 0.0010650, 0.0037693, 0.0011503, 0.0035818, -0.0019243, 0.0020173
4: -0.0016163, -0.0004664, -0.0015366, -0.0005026, -0.0008578, 0.0008183
5: -0.0149742, -0.0075016, -0.0144563, -0.0077371, -0.0055745, 0.0053174
6: 0.0034448, 0.0053415, 0.0035046, 0.0052100, -0.0013496, 0.0014149
7: 0.0057752, 0.0106823, 0.0059298, 0.0103422, -0.0034919, 0.0036607
8: 0.0034729, 0.0060536, 0.0035543, 0.0058747, -0.0018363, 0.0019251
9: -0.0088833, -0.0058909, -0.0086759, -0.0059852, -0.0022323, 0.0021293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010839, upper bound: 0.0014816
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010249, upper bound: 0.0014197
time: 1.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0016576, 1.0057553, -0.0035913, 0.0029876
1: -0.0009695, 0.0001375, -0.0008509, 0.0001701, -0.0008949, 0.0007444
2: -0.0107827, -0.0049161, -0.0109555, -0.0055445, -0.0039451, 0.0047423
3: 0.0009645, 0.0036347, 0.0012505, 0.0037134, -0.0021585, 0.0017957
4: -0.0015591, -0.0004236, -0.0015925, -0.0005452, -0.0007636, 0.0009179
5: -0.0146023, -0.0072237, -0.0148197, -0.0080141, -0.0049620, 0.0059645
6: 0.0033743, 0.0052471, 0.0035749, 0.0053022, -0.0015139, 0.0012594
7: 0.0055927, 0.0104381, 0.0061117, 0.0105809, -0.0039168, 0.0032584
8: 0.0033770, 0.0059252, 0.0036499, 0.0060002, -0.0020598, 0.0017136
9: -0.0087344, -0.0057796, -0.0088214, -0.0060961, -0.0019870, 0.0023885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010060, upper bound: 0.0014989
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0014292
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0016576, 1.0057553, -0.0033417, 0.0031150
1: -0.0009278, 0.0001933, -0.0008509, 0.0001701, -0.0008327, 0.0007762
2: -0.0110784, -0.0051370, -0.0109555, -0.0055445, -0.0041133, 0.0044126
3: 0.0010650, 0.0037693, 0.0012505, 0.0037134, -0.0020084, 0.0018722
4: -0.0016163, -0.0004664, -0.0015925, -0.0005452, -0.0007961, 0.0008541
5: -0.0149742, -0.0075016, -0.0148197, -0.0080141, -0.0051735, 0.0055499
6: 0.0034448, 0.0053415, 0.0035749, 0.0053022, -0.0014086, 0.0013131
7: 0.0057752, 0.0106823, 0.0061117, 0.0105809, -0.0036446, 0.0033974
8: 0.0034729, 0.0060536, 0.0036499, 0.0060002, -0.0019166, 0.0017866
9: -0.0088833, -0.0058909, -0.0088214, -0.0060961, -0.0020717, 0.0022224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008917, upper bound: 0.0011386
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008286, upper bound: 0.0013603
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0012008, 1.0055921, -0.0029381, 0.0031601
1: -0.0009370, 0.0001626, -0.0009647, 0.0001294, -0.0007321, 0.0007874
2: -0.0109158, -0.0050884, -0.0107400, -0.0049413, -0.0041728, 0.0038797
3: 0.0010429, 0.0036953, 0.0009760, 0.0036152, -0.0017659, 0.0018993
4: -0.0015848, -0.0004570, -0.0015508, -0.0004285, -0.0008076, 0.0007509
5: -0.0147698, -0.0074404, -0.0145486, -0.0072555, -0.0052483, 0.0048797
6: 0.0034293, 0.0052896, 0.0033823, 0.0052334, -0.0012385, 0.0013321
7: 0.0057350, 0.0105481, 0.0056135, 0.0104028, -0.0032044, 0.0034465
8: 0.0034518, 0.0059830, 0.0033880, 0.0059066, -0.0016852, 0.0018125
9: -0.0088014, -0.0058664, -0.0087129, -0.0057924, -0.0021017, 0.0019540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0015835
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0015304
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0013121, 1.0057251, -0.0030825, 0.0030825
1: -0.0009370, 0.0001626, -0.0009370, 0.0001626, -0.0007681, 0.0007681
2: -0.0109158, -0.0050884, -0.0109158, -0.0050884, -0.0040704, 0.0040704
3: 0.0010429, 0.0036953, 0.0010429, 0.0036953, -0.0018527, 0.0018527
4: -0.0015848, -0.0004570, -0.0015848, -0.0004570, -0.0007878, 0.0007878
5: -0.0147698, -0.0074404, -0.0147698, -0.0074404, -0.0051195, 0.0051195
6: 0.0034293, 0.0052896, 0.0034293, 0.0052896, -0.0012994, 0.0012994
7: 0.0057350, 0.0105481, 0.0057350, 0.0105481, -0.0033619, 0.0033619
8: 0.0034518, 0.0059830, 0.0034518, 0.0059830, -0.0017680, 0.0017680
9: -0.0088014, -0.0058664, -0.0088014, -0.0058664, -0.0020501, 0.0020501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0016376
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0016231
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0014889, 1.0059466, -0.0033631, 0.0029900
1: -0.0009370, 0.0001626, -0.0008930, 0.0002178, -0.0008380, 0.0007450
2: -0.0109158, -0.0050884, -0.0112082, -0.0053218, -0.0039483, 0.0044409
3: 0.0010429, 0.0036953, 0.0011491, 0.0038284, -0.0020213, 0.0017971
4: -0.0015848, -0.0004570, -0.0016414, -0.0005021, -0.0007642, 0.0008595
5: -0.0147698, -0.0074404, -0.0151375, -0.0077339, -0.0049659, 0.0055855
6: 0.0034293, 0.0052896, 0.0035038, 0.0053829, -0.0014176, 0.0012604
7: 0.0057350, 0.0105481, 0.0059277, 0.0107895, -0.0036679, 0.0032610
8: 0.0034518, 0.0059830, 0.0035532, 0.0061100, -0.0019289, 0.0017149
9: -0.0088014, -0.0058664, -0.0089487, -0.0059840, -0.0019885, 0.0022367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 10.07 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008778, upper bound: 0.0008485
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006878, upper bound: 0.0007426
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0011817, 1.0056244, -0.0030190, 0.0032562
1: -0.0009370, 0.0001626, -0.0009695, 0.0001375, -0.0007523, 0.0008114
2: -0.0109158, -0.0050884, -0.0107827, -0.0049161, -0.0042998, 0.0039866
3: 0.0010429, 0.0036953, 0.0009645, 0.0036347, -0.0018145, 0.0019571
4: -0.0015848, -0.0004570, -0.0015591, -0.0004236, -0.0008322, 0.0007716
5: -0.0147698, -0.0074404, -0.0146023, -0.0072237, -0.0054080, 0.0050141
6: 0.0034293, 0.0052896, 0.0033743, 0.0052471, -0.0012726, 0.0013726
7: 0.0057350, 0.0105481, 0.0055927, 0.0104381, -0.0032927, 0.0035513
8: 0.0034518, 0.0059830, 0.0033770, 0.0059252, -0.0017316, 0.0018676
9: -0.0088014, -0.0058664, -0.0087344, -0.0057796, -0.0021656, 0.0020079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012381, upper bound: 0.0014556
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012222, upper bound: 0.0013626
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0013490, 1.0058484, -0.0032985, 0.0031668
1: -0.0009370, 0.0001626, -0.0009278, 0.0001933, -0.0008219, 0.0007891
2: -0.0109158, -0.0050884, -0.0110784, -0.0051370, -0.0041818, 0.0043557
3: 0.0010429, 0.0036953, 0.0010650, 0.0037693, -0.0019825, 0.0019034
4: -0.0015848, -0.0004570, -0.0016163, -0.0004664, -0.0008094, 0.0008430
5: -0.0147698, -0.0074404, -0.0149742, -0.0075016, -0.0052596, 0.0054783
6: 0.0034293, 0.0052896, 0.0034448, 0.0053415, -0.0013904, 0.0013349
7: 0.0057350, 0.0105481, 0.0057752, 0.0106823, -0.0035975, 0.0034539
8: 0.0034518, 0.0059830, 0.0034729, 0.0060536, -0.0018919, 0.0018164
9: -0.0088014, -0.0058664, -0.0088833, -0.0058909, -0.0021062, 0.0021937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012381, upper bound: 0.0014556
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012222, upper bound: 0.0013626
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0014889, 1.0059466, 1.0011817, 1.0056244, -0.0029265, 0.0035382
1: -0.0008930, 0.0002178, -0.0009695, 0.0001375, -0.0007292, 0.0008816
2: -0.0112082, -0.0053218, -0.0107827, -0.0049161, -0.0046722, 0.0038644
3: 0.0011491, 0.0038284, 0.0009645, 0.0036347, -0.0017589, 0.0021266
4: -0.0016414, -0.0005021, -0.0015591, -0.0004236, -0.0009043, 0.0007479
5: -0.0151375, -0.0077339, -0.0146023, -0.0072237, -0.0058764, 0.0048604
6: 0.0035038, 0.0053829, 0.0033743, 0.0052471, -0.0012336, 0.0014915
7: 0.0059277, 0.0107895, 0.0055927, 0.0104381, -0.0031918, 0.0038589
8: 0.0035532, 0.0061100, 0.0033770, 0.0059252, -0.0016785, 0.0020294
9: -0.0089487, -0.0059840, -0.0087344, -0.0057796, -0.0023532, 0.0019463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0014889, 1.0059466, 1.0013490, 1.0058484, -0.0030411, 0.0033002
1: -0.0008930, 0.0002178, -0.0009278, 0.0001933, -0.0007578, 0.0008223
2: -0.0112082, -0.0053218, -0.0110784, -0.0051370, -0.0043579, 0.0040157
3: 0.0011491, 0.0038284, 0.0010650, 0.0037693, -0.0018278, 0.0019835
4: -0.0016414, -0.0005021, -0.0016163, -0.0004664, -0.0008435, 0.0007772
5: -0.0151375, -0.0077339, -0.0149742, -0.0075016, -0.0054810, 0.0050507
6: 0.0035038, 0.0053829, 0.0034448, 0.0053415, -0.0012819, 0.0013911
7: 0.0059277, 0.0107895, 0.0057752, 0.0106823, -0.0033167, 0.0035993
8: 0.0035532, 0.0061100, 0.0034729, 0.0060536, -0.0017442, 0.0018928
9: -0.0089487, -0.0059840, -0.0088833, -0.0058909, -0.0021948, 0.0020225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0011817, 1.0056244, -0.0030770, 0.0030770
1: -0.0009695, 0.0001375, -0.0009695, 0.0001375, -0.0007667, 0.0007667
2: -0.0107827, -0.0049161, -0.0107827, -0.0049161, -0.0040632, 0.0040632
3: 0.0009645, 0.0036347, 0.0009645, 0.0036347, -0.0018494, 0.0018494
4: -0.0015591, -0.0004236, -0.0015591, -0.0004236, -0.0007864, 0.0007864
5: -0.0146023, -0.0072237, -0.0146023, -0.0072237, -0.0051104, 0.0051104
6: 0.0033743, 0.0052471, 0.0033743, 0.0052471, -0.0012971, 0.0012971
7: 0.0055927, 0.0104381, 0.0055927, 0.0104381, -0.0033559, 0.0033559
8: 0.0033770, 0.0059252, 0.0033770, 0.0059252, -0.0017648, 0.0017648
9: -0.0087344, -0.0057796, -0.0087344, -0.0057796, -0.0020464, 0.0020464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014989
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013509, upper bound: 0.0014293
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0013490, 1.0058484, -0.0033598, 0.0029884
1: -0.0009695, 0.0001375, -0.0009278, 0.0001933, -0.0008372, 0.0007446
2: -0.0107827, -0.0049161, -0.0110784, -0.0051370, -0.0039461, 0.0044366
3: 0.0009645, 0.0036347, 0.0010650, 0.0037693, -0.0020193, 0.0017961
4: -0.0015591, -0.0004236, -0.0016163, -0.0004664, -0.0007638, 0.0008587
5: -0.0146023, -0.0072237, -0.0149742, -0.0075016, -0.0049632, 0.0055801
6: 0.0033743, 0.0052471, 0.0034448, 0.0053415, -0.0014163, 0.0012597
7: 0.0055927, 0.0104381, 0.0057752, 0.0106823, -0.0036643, 0.0032593
8: 0.0033770, 0.0059252, 0.0034729, 0.0060536, -0.0019270, 0.0017140
9: -0.0087344, -0.0057796, -0.0088833, -0.0058909, -0.0019875, 0.0022345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014989
time: 1.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013509, upper bound: 0.0014293
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0011817, 1.0056244, -0.0029884, 0.0033598
1: -0.0009278, 0.0001933, -0.0009695, 0.0001375, -0.0007446, 0.0008372
2: -0.0110784, -0.0051370, -0.0107827, -0.0049161, -0.0044366, 0.0039461
3: 0.0010650, 0.0037693, 0.0009645, 0.0036347, -0.0017961, 0.0020193
4: -0.0016163, -0.0004664, -0.0015591, -0.0004236, -0.0008587, 0.0007638
5: -0.0149742, -0.0075016, -0.0146023, -0.0072237, -0.0055801, 0.0049632
6: 0.0034448, 0.0053415, 0.0033743, 0.0052471, -0.0012597, 0.0014163
7: 0.0057752, 0.0106823, 0.0055927, 0.0104381, -0.0032593, 0.0036643
8: 0.0034729, 0.0060536, 0.0033770, 0.0059252, -0.0017140, 0.0019270
9: -0.0088833, -0.0058909, -0.0087344, -0.0057796, -0.0022345, 0.0019875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0013490, 1.0058484, -0.0031195, 0.0031195
1: -0.0009278, 0.0001933, -0.0009278, 0.0001933, -0.0007773, 0.0007773
2: -0.0110784, -0.0051370, -0.0110784, -0.0051370, -0.0041193, 0.0041193
3: 0.0010650, 0.0037693, 0.0010650, 0.0037693, -0.0018749, 0.0018749
4: -0.0016163, -0.0004664, -0.0016163, -0.0004664, -0.0007973, 0.0007973
5: -0.0149742, -0.0075016, -0.0149742, -0.0075016, -0.0051810, 0.0051810
6: 0.0034448, 0.0053415, 0.0034448, 0.0053415, -0.0013150, 0.0013150
7: 0.0057752, 0.0106823, 0.0057752, 0.0106823, -0.0034023, 0.0034023
8: 0.0034729, 0.0060536, 0.0034729, 0.0060536, -0.0017892, 0.0017892
9: -0.0088833, -0.0058909, -0.0088833, -0.0058909, -0.0020747, 0.0020747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
time: 1.15 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.59 seconds
IS_A1_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0014786, upper bound: 0.0015052
IS_A1_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0014670, upper bound: 0.0014429
IS_A1_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0014786, upper bound: 0.0015394
IS_A1_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0014670, upper bound: 0.0015165
IS_A2_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0015835
IS_A2_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0015304
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0016376
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0016230
IS_A2_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0009886, upper bound: 0.0009878
IS_A2_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0008009, upper bound: 0.0009020
IS_A2_B1_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0016163
IS_A2_B1_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0015725
IS_A2_B1_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0016601
IS_A2_B1_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0016494
IS_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0009903, upper bound: 0.0010194
IS_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0008122, upper bound: 0.0009449
IS_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
IS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
IS_A2_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0016483
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0014496, upper bound: 0.0016395
IS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0010839, upper bound: 0.0014816
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0010249, upper bound: 0.0014197
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0010060, upper bound: 0.0014989
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0014292
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0008917, upper bound: 0.0011386
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0008286, upper bound: 0.0013603
IS_A2_B2_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0015835
IS_A2_B2_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0015304
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013776, upper bound: 0.0016376
IS_A2_B2_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0016231
IS_A2_B2_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0008778, upper bound: 0.0008485
IS_A2_B2_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0006878, upper bound: 0.0007426
IS_A2_B2_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0012381, upper bound: 0.0014556
IS_A2_B2_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0012222, upper bound: 0.0013626
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0012381, upper bound: 0.0014556
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0012222, upper bound: 0.0013626
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
IS_A2_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014989
IS_A2_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013509, upper bound: 0.0014293
IS_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014989
IS_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013509, upper bound: 0.0014293
IS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
IS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.59
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0013394, 1.0057197, 1.0016202, 1.0056431, -0.0032869, 0.0030720
1: -0.0009302, 0.0001612, -0.0008602, 0.0001422, -0.0008190, 0.0007655
2: -0.0109085, -0.0051244, -0.0108075, -0.0054952, -0.0040566, 0.0043404
3: 0.0010593, 0.0036919, 0.0012280, 0.0036460, -0.0019755, 0.0018464
4: -0.0015834, -0.0004639, -0.0015639, -0.0005357, -0.0007851, 0.0008401
5: -0.0147605, -0.0074857, -0.0146335, -0.0079520, -0.0051021, 0.0054590
6: 0.0034408, 0.0052872, 0.0035591, 0.0052550, -0.0013856, 0.0012950
7: 0.0057647, 0.0105420, 0.0060709, 0.0104586, -0.0035849, 0.0033505
8: 0.0034675, 0.0059798, 0.0036285, 0.0059359, -0.0018853, 0.0017620
9: -0.0087977, -0.0058846, -0.0087468, -0.0060713, -0.0020431, 0.0021860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016230
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013813, 1.0059683, 1.0016456, 1.0056387, -0.0033010, 0.0033390
1: -0.0009198, 0.0002232, -0.0008539, 0.0001411, -0.0008225, 0.0008320
2: -0.0112368, -0.0051796, -0.0108016, -0.0055287, -0.0044091, 0.0043589
3: 0.0010844, 0.0038414, 0.0012433, 0.0036433, -0.0019840, 0.0020068
4: -0.0016470, -0.0004746, -0.0015627, -0.0005422, -0.0008534, 0.0008437
5: -0.0151735, -0.0075551, -0.0146261, -0.0079942, -0.0055455, 0.0054824
6: 0.0034584, 0.0053920, 0.0035698, 0.0052531, -0.0013915, 0.0014075
7: 0.0058103, 0.0108132, 0.0060986, 0.0104537, -0.0036002, 0.0036417
8: 0.0034914, 0.0061224, 0.0036431, 0.0059334, -0.0018933, 0.0019151
9: -0.0089631, -0.0059123, -0.0087439, -0.0060882, -0.0022207, 0.0021954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016206
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016230
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0013394, 1.0057197, 1.0014908, 1.0055364, -0.0032242, 0.0032567
1: -0.0009302, 0.0001612, -0.0008925, 0.0001156, -0.0008034, 0.0008115
2: -0.0109085, -0.0051244, -0.0106665, -0.0053243, -0.0043004, 0.0042575
3: 0.0010593, 0.0036919, 0.0011503, 0.0035818, -0.0019378, 0.0019574
4: -0.0015834, -0.0004639, -0.0015366, -0.0005026, -0.0008323, 0.0008240
5: -0.0147605, -0.0074857, -0.0144563, -0.0077371, -0.0054088, 0.0053548
6: 0.0034408, 0.0052872, 0.0035046, 0.0052100, -0.0013591, 0.0013728
7: 0.0057647, 0.0105420, 0.0059298, 0.0103422, -0.0035164, 0.0035519
8: 0.0034675, 0.0059798, 0.0035543, 0.0058747, -0.0018493, 0.0018679
9: -0.0087977, -0.0058846, -0.0086759, -0.0059852, -0.0021659, 0.0021443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016462
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016494
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013813, 1.0059683, 1.0015188, 1.0055326, -0.0032386, 0.0035092
1: -0.0009198, 0.0002232, -0.0008855, 0.0001146, -0.0008070, 0.0008744
2: -0.0112368, -0.0051796, -0.0106613, -0.0053613, -0.0046339, 0.0042766
3: 0.0010844, 0.0038414, 0.0011671, 0.0035795, -0.0019465, 0.0021091
4: -0.0016470, -0.0004746, -0.0015356, -0.0005098, -0.0008969, 0.0008277
5: -0.0151735, -0.0075551, -0.0144497, -0.0077837, -0.0058282, 0.0053788
6: 0.0034584, 0.0053920, 0.0035164, 0.0052083, -0.0013652, 0.0014793
7: 0.0058103, 0.0108132, 0.0059604, 0.0103379, -0.0035322, 0.0038273
8: 0.0034914, 0.0061224, 0.0035704, 0.0058724, -0.0018575, 0.0020127
9: -0.0089631, -0.0059123, -0.0086732, -0.0060039, -0.0023339, 0.0021539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016462
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016494
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0012124, 1.0056196, 1.0014908, 1.0055364, -0.0032764, 0.0030761
1: -0.0009619, 0.0001363, -0.0008925, 0.0001156, -0.0008164, 0.0007665
2: -0.0107765, -0.0049566, -0.0106665, -0.0053243, -0.0040619, 0.0043265
3: 0.0009829, 0.0036319, 0.0011503, 0.0035818, -0.0019692, 0.0018488
4: -0.0015579, -0.0004314, -0.0015366, -0.0005026, -0.0007862, 0.0008374
5: -0.0145946, -0.0072746, -0.0144563, -0.0077371, -0.0051088, 0.0054416
6: 0.0033872, 0.0052451, 0.0035046, 0.0052100, -0.0013811, 0.0012967
7: 0.0056261, 0.0104330, 0.0059298, 0.0103422, -0.0035734, 0.0033549
8: 0.0033946, 0.0059225, 0.0035543, 0.0058747, -0.0018792, 0.0017643
9: -0.0087313, -0.0058000, -0.0086759, -0.0059852, -0.0020458, 0.0021790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016341
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016395
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0012611, 1.0058589, 1.0015188, 1.0055326, -0.0032659, 0.0033217
1: -0.0009497, 0.0001959, -0.0008855, 0.0001146, -0.0008138, 0.0008277
2: -0.0110923, -0.0050209, -0.0106613, -0.0053613, -0.0043862, 0.0043125
3: 0.0010122, 0.0037756, 0.0011671, 0.0035795, -0.0019629, 0.0019964
4: -0.0016190, -0.0004439, -0.0015356, -0.0005098, -0.0008489, 0.0008347
5: -0.0149918, -0.0073556, -0.0144497, -0.0077837, -0.0055167, 0.0054240
6: 0.0034078, 0.0053459, 0.0035164, 0.0052083, -0.0013767, 0.0014002
7: 0.0056793, 0.0106939, 0.0059604, 0.0103379, -0.0035619, 0.0036228
8: 0.0034225, 0.0060597, 0.0035704, 0.0058724, -0.0018732, 0.0019052
9: -0.0088903, -0.0058324, -0.0086732, -0.0060039, -0.0022091, 0.0021720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015382, upper bound: 0.0016341
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015382, upper bound: 0.0016395
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0013394, 1.0057197, 1.0013121, 1.0057251, -0.0030531, 0.0030795
1: -0.0009302, 0.0001612, -0.0009370, 0.0001626, -0.0007607, 0.0007673
2: -0.0109085, -0.0051244, -0.0109158, -0.0050884, -0.0040665, 0.0040315
3: 0.0010593, 0.0036919, 0.0010429, 0.0036953, -0.0018350, 0.0018509
4: -0.0015834, -0.0004639, -0.0015848, -0.0004570, -0.0007871, 0.0007803
5: -0.0147605, -0.0074857, -0.0147698, -0.0074404, -0.0051146, 0.0050706
6: 0.0034408, 0.0052872, 0.0034293, 0.0052896, -0.0012870, 0.0012981
7: 0.0057647, 0.0105420, 0.0057350, 0.0105481, -0.0033298, 0.0033587
8: 0.0034675, 0.0059798, 0.0034518, 0.0059830, -0.0017511, 0.0017663
9: -0.0087977, -0.0058846, -0.0088014, -0.0058664, -0.0020481, 0.0020305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016231
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013813, 1.0059683, 1.0013386, 1.0057207, -0.0030623, 0.0033701
1: -0.0009198, 0.0002232, -0.0009304, 0.0001615, -0.0007630, 0.0008397
2: -0.0112368, -0.0051796, -0.0109099, -0.0051234, -0.0044501, 0.0040438
3: 0.0010844, 0.0038414, 0.0010588, 0.0036926, -0.0018405, 0.0020255
4: -0.0016470, -0.0004746, -0.0015837, -0.0004637, -0.0008613, 0.0007827
5: -0.0151735, -0.0075551, -0.0147623, -0.0074844, -0.0055971, 0.0050860
6: 0.0034584, 0.0053920, 0.0034405, 0.0052877, -0.0012909, 0.0014206
7: 0.0058103, 0.0108132, 0.0057639, 0.0105432, -0.0033399, 0.0036755
8: 0.0034914, 0.0061224, 0.0034670, 0.0059804, -0.0017564, 0.0019329
9: -0.0089631, -0.0059123, -0.0087984, -0.0058840, -0.0022413, 0.0020367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016206
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016230
time: 1.36 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.00 seconds
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016230
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016206
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016230
IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016462
IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016494
IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016462
IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016494
IS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016341
IS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016395
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015382, upper bound: 0.0016341
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015382, upper bound: 0.0016395
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016231
IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016206
IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.00
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016230

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0013394, 1.0057197, 1.0016490, 1.0056376, -0.0032838, 0.0030421
1: -0.0009302, 0.0001612, -0.0008531, 0.0001408, -0.0008182, 0.0007580
2: -0.0109085, -0.0051244, -0.0108002, -0.0055333, -0.0040171, 0.0043363
3: 0.0010593, 0.0036919, 0.0012454, 0.0036426, -0.0019737, 0.0018284
4: -0.0015834, -0.0004639, -0.0015625, -0.0005431, -0.0007775, 0.0008393
5: -0.0147605, -0.0074857, -0.0146243, -0.0080000, -0.0050524, 0.0054539
6: 0.0034408, 0.0052872, 0.0035713, 0.0052526, -0.0013842, 0.0012824
7: 0.0057647, 0.0105420, 0.0061024, 0.0104526, -0.0035815, 0.0033179
8: 0.0034675, 0.0059798, 0.0036451, 0.0059328, -0.0018835, 0.0017448
9: -0.0087977, -0.0058846, -0.0087432, -0.0060905, -0.0020232, 0.0021840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 8.14 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011673, upper bound: 0.0015017
time: 1.48 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013442, upper bound: 0.0014839
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013394, 1.0057197, 1.0016844, 1.0059094, -0.0036023, 0.0030099
1: -0.0009302, 0.0001612, -0.0008442, 0.0002085, -0.0008976, 0.0007500
2: -0.0109085, -0.0051244, -0.0111590, -0.0055800, -0.0039745, 0.0047568
3: 0.0010593, 0.0036919, 0.0012667, 0.0038060, -0.0021651, 0.0018090
4: -0.0015834, -0.0004639, -0.0016319, -0.0005521, -0.0007693, 0.0009207
5: -0.0147605, -0.0074857, -0.0150756, -0.0080588, -0.0049988, 0.0059828
6: 0.0034408, 0.0052872, 0.0035862, 0.0053672, -0.0015185, 0.0012688
7: 0.0057647, 0.0105420, 0.0061410, 0.0107489, -0.0039288, 0.0032827
8: 0.0034675, 0.0059798, 0.0036654, 0.0060886, -0.0020661, 0.0017263
9: -0.0087977, -0.0058846, -0.0089239, -0.0061140, -0.0020018, 0.0023958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 8.18 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011673, upper bound: 0.0015017
time: 1.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013442, upper bound: 0.0014839
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 1.0013813, 1.0059683, 1.0016490, 1.0056376, -0.0032401, 0.0033337
1: -0.0009198, 0.0002232, -0.0008531, 0.0001408, -0.0008074, 0.0008307
2: -0.0112368, -0.0051796, -0.0108002, -0.0055333, -0.0044021, 0.0042786
3: 0.0010844, 0.0038414, 0.0012454, 0.0036426, -0.0019474, 0.0020037
4: -0.0016470, -0.0004746, -0.0015625, -0.0005431, -0.0008520, 0.0008281
5: -0.0151735, -0.0075551, -0.0146243, -0.0080000, -0.0055367, 0.0053813
6: 0.0034584, 0.0053920, 0.0035713, 0.0052526, -0.0013658, 0.0014053
7: 0.0058103, 0.0108132, 0.0061024, 0.0104526, -0.0035338, 0.0036359
8: 0.0034914, 0.0061224, 0.0036451, 0.0059328, -0.0018584, 0.0019121
9: -0.0089631, -0.0059123, -0.0087432, -0.0060905, -0.0022171, 0.0021549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 8.11 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011752, upper bound: 0.0014692
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013459, upper bound: 0.0014354
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013813, 1.0059683, 1.0016844, 1.0059094, -0.0035044, 0.0032441
1: -0.0009198, 0.0002232, -0.0008442, 0.0002085, -0.0008732, 0.0008084
2: -0.0112368, -0.0051796, -0.0111590, -0.0055800, -0.0042839, 0.0046275
3: 0.0010844, 0.0038414, 0.0012667, 0.0038060, -0.0021063, 0.0019498
4: -0.0016470, -0.0004746, -0.0016319, -0.0005521, -0.0008291, 0.0008957
5: -0.0151735, -0.0075551, -0.0150756, -0.0080588, -0.0053880, 0.0058202
6: 0.0034584, 0.0053920, 0.0035862, 0.0053672, -0.0014772, 0.0013675
7: 0.0058103, 0.0108132, 0.0061410, 0.0107489, -0.0038221, 0.0035382
8: 0.0034914, 0.0061224, 0.0036654, 0.0060886, -0.0020100, 0.0018607
9: -0.0089631, -0.0059123, -0.0089239, -0.0061140, -0.0021576, 0.0023307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 8.06 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011752, upper bound: 0.0014692
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013459, upper bound: 0.0014354
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0013394, 1.0057197, 1.0015237, 1.0055317, -0.0032216, 0.0032191
1: -0.0009302, 0.0001612, -0.0008843, 0.0001144, -0.0008027, 0.0008021
2: -0.0109085, -0.0051244, -0.0106602, -0.0053677, -0.0042508, 0.0042541
3: 0.0010593, 0.0036919, 0.0011700, 0.0035789, -0.0019363, 0.0019348
4: -0.0015834, -0.0004639, -0.0015354, -0.0005110, -0.0008227, 0.0008234
5: -0.0147605, -0.0074857, -0.0144482, -0.0077917, -0.0053464, 0.0053506
6: 0.0034408, 0.0052872, 0.0035184, 0.0052079, -0.0013580, 0.0013570
7: 0.0057647, 0.0105420, 0.0059657, 0.0103369, -0.0035137, 0.0035109
8: 0.0034675, 0.0059798, 0.0035731, 0.0058719, -0.0018478, 0.0018463
9: -0.0087977, -0.0058846, -0.0086727, -0.0060071, -0.0021409, 0.0021426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 8.06 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 145

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011673, upper bound: 0.0015249
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013442, upper bound: 0.0015138
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013394, 1.0057197, 1.0015628, 1.0057992, -0.0035369, 0.0031702
1: -0.0009302, 0.0001612, -0.0008745, 0.0001811, -0.0008813, 0.0007899
2: -0.0109085, -0.0051244, -0.0110135, -0.0054194, -0.0041862, 0.0046704
3: 0.0010593, 0.0036919, 0.0011936, 0.0037398, -0.0021258, 0.0019054
4: -0.0015834, -0.0004639, -0.0016038, -0.0005210, -0.0008102, 0.0009040
5: -0.0147605, -0.0074857, -0.0148927, -0.0078567, -0.0052652, 0.0058742
6: 0.0034408, 0.0052872, 0.0035350, 0.0053208, -0.0014909, 0.0013364
7: 0.0057647, 0.0105420, 0.0060084, 0.0106288, -0.0038575, 0.0034575
8: 0.0034675, 0.0059798, 0.0035956, 0.0060254, -0.0020286, 0.0018183
9: -0.0087977, -0.0058846, -0.0088506, -0.0060331, -0.0021084, 0.0023523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242

Time for candidate selection: 8.10 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.25 + 597.99 = 601.24 seconds
