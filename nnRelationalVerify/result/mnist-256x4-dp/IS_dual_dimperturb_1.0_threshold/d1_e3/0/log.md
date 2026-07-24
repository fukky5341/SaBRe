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
execution time: IAR + RelationalAnalysis = 1.16 + 2.01 = 3.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0020250, upper bound: 0.0020250

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019058, upper bound: 0.0018018
time: 1.18 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019058, upper bound: 0.0019058
time: 1.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.40 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.40
Output dim: 0, lower bound: -0.0019058, upper bound: 0.0018018
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.40
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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018018, upper bound: 0.0018018
time: 1.14 seconds

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018332, upper bound: 0.0018126
time: 1.15 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018348, upper bound: 0.0018348
time: 1.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -0.0018018, upper bound: 0.0018018
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -0.0018018, upper bound: 0.0018018
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -0.0018332, upper bound: 0.0018126
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -0.0018348, upper bound: 0.0018348

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

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017238, upper bound: 0.0017284
time: 1.16 seconds

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

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

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
0: 1.0010930, 1.0056691, 1.0015438, 1.0059171, -0.0039117, 0.0031020
1: -0.0009916, 0.0001486, -0.0008793, 0.0002104, -0.0009747, 0.0007729
2: -0.0108418, -0.0047990, -0.0111691, -0.0053943, -0.0040962, 0.0051654
3: 0.0009112, 0.0036616, 0.0011821, 0.0038106, -0.0023511, 0.0018644
4: -0.0015705, -0.0004010, -0.0016339, -0.0005162, -0.0007928, 0.0009998
5: -0.0146766, -0.0070765, -0.0150883, -0.0078251, -0.0051519, 0.0064967
6: 0.0033369, 0.0052659, 0.0035269, 0.0053704, -0.0016489, 0.0013076
7: 0.0054960, 0.0104869, 0.0059876, 0.0107573, -0.0042663, 0.0033832
8: 0.0033261, 0.0059508, 0.0035847, 0.0060930, -0.0022436, 0.0017792
9: -0.0087641, -0.0057207, -0.0089290, -0.0060205, -0.0020630, 0.0026016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018126
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018126
time: 1.50 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 1.0010248, 1.0056758, 1.0014126, 1.0058177, -0.0039487, 0.0031816
1: -0.0010086, 0.0001503, -0.0009120, 0.0001856, -0.0009839, 0.0007928
2: -0.0108505, -0.0047090, -0.0110378, -0.0052210, -0.0042013, 0.0052142
3: 0.0008702, 0.0036655, 0.0011032, 0.0037508, -0.0023733, 0.0019122
4: -0.0015722, -0.0003835, -0.0016085, -0.0004826, -0.0008131, 0.0010092
5: -0.0146876, -0.0069632, -0.0149232, -0.0076072, -0.0052841, 0.0065580
6: 0.0033082, 0.0052687, 0.0034716, 0.0053285, -0.0016645, 0.0013412
7: 0.0054216, 0.0104941, 0.0058445, 0.0106489, -0.0043066, 0.0034700
8: 0.0032870, 0.0059546, 0.0035094, 0.0060360, -0.0022648, 0.0018248
9: -0.0087685, -0.0056753, -0.0088629, -0.0059332, -0.0021160, 0.0026261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017290, upper bound: 0.0018348
time: 1.12 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017290, upper bound: 0.0018348
time: 1.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.48 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -0.0017238, upper bound: 0.0017284
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -0.0017427, upper bound: 0.0017290
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -0.0017238, upper bound: 0.0017284
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -0.0017427, upper bound: 0.0017290
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018126
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -0.0017284, upper bound: 0.0018126
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 0, lower bound: -0.0017290, upper bound: 0.0018348
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.48
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

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017313
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017496
time: 1.13 seconds

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

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
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018127, upper bound: 0.0017125
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018127, upper bound: 0.0017284
time: 1.10 seconds

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018331, upper bound: 0.0017126
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018331, upper bound: 0.0017290
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 1.0010930, 1.0056691, 1.0015463, 1.0056882, -0.0036481, 0.0031462
1: -0.0009916, 0.0001486, -0.0008787, 0.0001534, -0.0009090, 0.0007839
2: -0.0108418, -0.0047990, -0.0108669, -0.0053976, -0.0041545, 0.0048173
3: 0.0009112, 0.0036616, 0.0011836, 0.0036730, -0.0021926, 0.0018910
4: -0.0015705, -0.0004010, -0.0015754, -0.0005168, -0.0008041, 0.0009324
5: -0.0146766, -0.0070765, -0.0147082, -0.0078293, -0.0052253, 0.0060589
6: 0.0033369, 0.0052659, 0.0035280, 0.0052739, -0.0015378, 0.0013262
7: 0.0054960, 0.0104869, 0.0059904, 0.0105077, -0.0039788, 0.0034314
8: 0.0033261, 0.0059508, 0.0035861, 0.0059617, -0.0020924, 0.0018045
9: -0.0087641, -0.0057207, -0.0087768, -0.0060222, -0.0020924, 0.0024263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018126
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018126
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 1.0010930, 1.0056691, 1.0012380, 1.0057670, -0.0033489, 0.0031279
1: -0.0009916, 0.0001486, -0.0009555, 0.0001730, -0.0008344, 0.0007794
2: -0.0108418, -0.0047990, -0.0109709, -0.0049904, -0.0041304, 0.0044221
3: 0.0009112, 0.0036616, 0.0009983, 0.0037204, -0.0020128, 0.0018800
4: -0.0015705, -0.0004010, -0.0015955, -0.0004380, -0.0007994, 0.0008559
5: -0.0146766, -0.0070765, -0.0148391, -0.0073172, -0.0051949, 0.0055619
6: 0.0033369, 0.0052659, 0.0033980, 0.0053072, -0.0014117, 0.0013185
7: 0.0054960, 0.0104869, 0.0056541, 0.0105936, -0.0036524, 0.0034114
8: 0.0033261, 0.0059508, 0.0034093, 0.0060069, -0.0019208, 0.0017940
9: -0.0087641, -0.0057207, -0.0088292, -0.0058171, -0.0020803, 0.0022272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018126
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018126
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 1.0010248, 1.0056758, 1.0014151, 1.0055836, -0.0036919, 0.0032477
1: -0.0010086, 0.0001503, -0.0009113, 0.0001273, -0.0009199, 0.0008092
2: -0.0108505, -0.0047090, -0.0107289, -0.0052244, -0.0042885, 0.0048752
3: 0.0008702, 0.0036655, 0.0011048, 0.0036102, -0.0022190, 0.0019520
4: -0.0015722, -0.0003835, -0.0015487, -0.0004833, -0.0008300, 0.0009436
5: -0.0146876, -0.0069632, -0.0145347, -0.0076115, -0.0053939, 0.0061317
6: 0.0033082, 0.0052687, 0.0034727, 0.0052299, -0.0015563, 0.0013690
7: 0.0054216, 0.0104941, 0.0058473, 0.0103937, -0.0040266, 0.0035421
8: 0.0032870, 0.0059546, 0.0035109, 0.0059018, -0.0021175, 0.0018627
9: -0.0087685, -0.0056753, -0.0087073, -0.0059349, -0.0021599, 0.0024554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018331
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018162
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 1.0010248, 1.0056758, 1.0011046, 1.0056679, -0.0034027, 0.0032061
1: -0.0010086, 0.0001503, -0.0009887, 0.0001483, -0.0008479, 0.0007989
2: -0.0108505, -0.0047090, -0.0108400, -0.0048142, -0.0042337, 0.0044932
3: 0.0008702, 0.0036655, 0.0009181, 0.0036608, -0.0020451, 0.0019270
4: -0.0015722, -0.0003835, -0.0015702, -0.0004039, -0.0008194, 0.0008697
5: -0.0146876, -0.0069632, -0.0146744, -0.0070956, -0.0053248, 0.0056513
6: 0.0033082, 0.0052687, 0.0033418, 0.0052654, -0.0014344, 0.0013515
7: 0.0054216, 0.0104941, 0.0055086, 0.0104855, -0.0037111, 0.0034967
8: 0.0032870, 0.0059546, 0.0033328, 0.0059501, -0.0019516, 0.0018389
9: -0.0087685, -0.0056753, -0.0087632, -0.0057283, -0.0021323, 0.0022630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018332
time: 1.12 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018162
time: 1.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.02 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017313
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017496
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017496, upper bound: 0.0017313
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017496, upper bound: 0.0017502
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0018127, upper bound: 0.0017125
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0018127, upper bound: 0.0017284
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0018331, upper bound: 0.0017126
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0018331, upper bound: 0.0017290
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018126
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018126
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018126
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018126
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018331
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018162
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018332
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.02
Output dim: 0, lower bound: -0.0017125, upper bound: 0.0018162

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015997, upper bound: 0.0015998
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015924, upper bound: 0.0015924
time: 1.05 seconds

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016735, upper bound: 0.0017076
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016899, upper bound: 0.0017082
time: 1.21 seconds

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017076, upper bound: 0.0016735
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017082, upper bound: 0.0016899
time: 1.22 seconds

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017076, upper bound: 0.0016771
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017082, upper bound: 0.0016918
time: 1.20 seconds

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016575, upper bound: 0.0015746
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015634
time: 1.18 seconds

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
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017431, upper bound: 0.0016822
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017748, upper bound: 0.0016840
time: 1.09 seconds

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

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017926, upper bound: 0.0016542
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017962, upper bound: 0.0016694
time: 1.13 seconds

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017625, upper bound: 0.0016700
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017962, upper bound: 0.0016715
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B1_A1

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016575
time: 1.21 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_B1_A2

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016672, upper bound: 0.0017431
time: 1.35 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016694, upper bound: 0.0017748
time: 1.50 seconds

## BFS IS instance: IS_A2_B1_B2_A1

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
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016575
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_B2_A2

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016672, upper bound: 0.0017431
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016694, upper bound: 0.0017748
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B1_A1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016542, upper bound: 0.0017927
time: 1.11 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016694, upper bound: 0.0017962
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_B1_A2

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016672, upper bound: 0.0017501
time: 1.61 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016694, upper bound: 0.0017784
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_B2_A1

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016542, upper bound: 0.0017927
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016695, upper bound: 0.0017962
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B2_A2

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016672, upper bound: 0.0017501
time: 1.13 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016695, upper bound: 0.0017784
time: 1.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.41 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0015997, upper bound: 0.0015998
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0015924, upper bound: 0.0015924
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016735, upper bound: 0.0017076
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016899, upper bound: 0.0017082
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017076, upper bound: 0.0016735
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017082, upper bound: 0.0016899
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017076, upper bound: 0.0016771
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017082, upper bound: 0.0016918
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016575, upper bound: 0.0015746
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016355, upper bound: 0.0015634
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017431, upper bound: 0.0016822
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017748, upper bound: 0.0016840
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017926, upper bound: 0.0016542
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017962, upper bound: 0.0016694
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017625, upper bound: 0.0016700
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0017962, upper bound: 0.0016715
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016575
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016672, upper bound: 0.0017431
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016694, upper bound: 0.0017748
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0015746, upper bound: 0.0016575
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016355
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016672, upper bound: 0.0017431
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016694, upper bound: 0.0017748
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016542, upper bound: 0.0017927
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016694, upper bound: 0.0017962
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016672, upper bound: 0.0017501
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016694, upper bound: 0.0017784
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016542, upper bound: 0.0017927
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016695, upper bound: 0.0017962
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016672, upper bound: 0.0017501
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -0.0016695, upper bound: 0.0017784

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 1.0015520, 1.0056291, 1.0012932, 1.0054303, -0.0028490, 0.0032857
1: -0.0008772, 0.0001386, -0.0009417, 0.0000892, -0.0007099, 0.0008187
2: -0.0107887, -0.0054051, -0.0105265, -0.0050633, -0.0043387, 0.0037621
3: 0.0011870, 0.0036374, 0.0010315, 0.0035181, -0.0017124, 0.0019748
4: -0.0015602, -0.0005183, -0.0015095, -0.0004521, -0.0008397, 0.0007282
5: -0.0146099, -0.0078388, -0.0142801, -0.0074088, -0.0054569, 0.0047318
6: 0.0035304, 0.0052490, 0.0034213, 0.0051653, -0.0012010, 0.0013850
7: 0.0059966, 0.0104431, 0.0057142, 0.0102265, -0.0031073, 0.0035835
8: 0.0035894, 0.0059278, 0.0034409, 0.0058139, -0.0016341, 0.0018845
9: -0.0087374, -0.0060259, -0.0086053, -0.0058538, -0.0021852, 0.0018948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0015576
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014594, upper bound: 0.0015474
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0014200, 1.0055397, -0.0029899, 0.0032523
1: -0.0008787, 0.0001534, -0.0009101, 0.0001164, -0.0007450, 0.0008104
2: -0.0108669, -0.0053976, -0.0106708, -0.0052308, -0.0042946, 0.0039481
3: 0.0011836, 0.0036730, 0.0011077, 0.0035838, -0.0017970, 0.0019547
4: -0.0015754, -0.0005168, -0.0015374, -0.0004845, -0.0008312, 0.0007641
5: -0.0147082, -0.0078293, -0.0144616, -0.0076196, -0.0054015, 0.0049656
6: 0.0035280, 0.0052739, 0.0034748, 0.0052113, -0.0012603, 0.0013710
7: 0.0059904, 0.0105077, 0.0058526, 0.0103457, -0.0032609, 0.0035471
8: 0.0035861, 0.0059617, 0.0035137, 0.0058766, -0.0017149, 0.0018654
9: -0.0087768, -0.0060222, -0.0086780, -0.0059382, -0.0021630, 0.0019885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015587, upper bound: 0.0015797
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015505, upper bound: 0.0015761
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0012932, 1.0054303, 1.0015520, 1.0056291, -0.0032857, 0.0028490
1: -0.0009417, 0.0000892, -0.0008772, 0.0001386, -0.0008187, 0.0007099
2: -0.0105265, -0.0050633, -0.0107887, -0.0054051, -0.0037621, 0.0043387
3: 0.0010315, 0.0035181, 0.0011870, 0.0036374, -0.0019748, 0.0017124
4: -0.0015095, -0.0004521, -0.0015602, -0.0005183, -0.0007282, 0.0008397
5: -0.0142801, -0.0074088, -0.0146099, -0.0078388, -0.0047318, 0.0054569
6: 0.0034213, 0.0051653, 0.0035304, 0.0052490, -0.0013850, 0.0012010
7: 0.0057142, 0.0102265, 0.0059966, 0.0104431, -0.0035835, 0.0031073
8: 0.0034409, 0.0058139, 0.0035894, 0.0059278, -0.0018845, 0.0016341
9: -0.0086053, -0.0058538, -0.0087374, -0.0060259, -0.0018948, 0.0021852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015576, upper bound: 0.0014916
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015474, upper bound: 0.0014595
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0014200, 1.0055397, 1.0015463, 1.0056882, -0.0032523, 0.0029898
1: -0.0009101, 0.0001164, -0.0008787, 0.0001534, -0.0008104, 0.0007450
2: -0.0106708, -0.0052308, -0.0108669, -0.0053976, -0.0039481, 0.0042946
3: 0.0011077, 0.0035838, 0.0011836, 0.0036730, -0.0019547, 0.0017970
4: -0.0015374, -0.0004845, -0.0015754, -0.0005168, -0.0007641, 0.0008312
5: -0.0144616, -0.0076196, -0.0147082, -0.0078293, -0.0049656, 0.0054015
6: 0.0034748, 0.0052113, 0.0035280, 0.0052739, -0.0013710, 0.0012603
7: 0.0058526, 0.0103457, 0.0059904, 0.0105077, -0.0035471, 0.0032609
8: 0.0035137, 0.0058766, 0.0035861, 0.0059617, -0.0018654, 0.0017149
9: -0.0086780, -0.0059382, -0.0087768, -0.0060222, -0.0019885, 0.0021630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015797, upper bound: 0.0015587
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015761, upper bound: 0.0015505
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 1.0012932, 1.0054303, 1.0014273, 1.0055231, -0.0031017, 0.0029198
1: -0.0009417, 0.0000892, -0.0009083, 0.0001122, -0.0007729, 0.0007275
2: -0.0105265, -0.0050633, -0.0106489, -0.0052404, -0.0038556, 0.0040958
3: 0.0010315, 0.0035181, 0.0011121, 0.0035738, -0.0018642, 0.0017549
4: -0.0015095, -0.0004521, -0.0015332, -0.0004864, -0.0007462, 0.0007927
5: -0.0142801, -0.0074088, -0.0144340, -0.0076316, -0.0048494, 0.0051514
6: 0.0034213, 0.0051653, 0.0034778, 0.0052043, -0.0013075, 0.0012308
7: 0.0057142, 0.0102265, 0.0058605, 0.0103276, -0.0033828, 0.0031845
8: 0.0034409, 0.0058139, 0.0035178, 0.0058670, -0.0017790, 0.0016747
9: -0.0086053, -0.0058538, -0.0086670, -0.0059430, -0.0019419, 0.0020628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015843, upper bound: 0.0015475
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015842, upper bound: 0.0015420
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0014200, 1.0055397, 1.0014151, 1.0055836, -0.0030695, 0.0030472
1: -0.0009101, 0.0001164, -0.0009113, 0.0001273, -0.0007648, 0.0007593
2: -0.0106708, -0.0052308, -0.0107289, -0.0052244, -0.0040238, 0.0040533
3: 0.0011077, 0.0035838, 0.0011048, 0.0036102, -0.0018449, 0.0018315
4: -0.0015374, -0.0004845, -0.0015487, -0.0004833, -0.0007788, 0.0007845
5: -0.0144616, -0.0076196, -0.0145347, -0.0076115, -0.0050609, 0.0050979
6: 0.0034748, 0.0052113, 0.0034727, 0.0052299, -0.0012939, 0.0012845
7: 0.0058526, 0.0103457, 0.0058473, 0.0103937, -0.0033477, 0.0033234
8: 0.0035137, 0.0058766, 0.0035109, 0.0059018, -0.0017605, 0.0017478
9: -0.0086780, -0.0059382, -0.0087073, -0.0059349, -0.0020266, 0.0020414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015915, upper bound: 0.0015781
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015943, upper bound: 0.0015779
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0013096, 1.0057642, -0.0032071, 0.0033875
1: -0.0008787, 0.0001534, -0.0009376, 0.0001723, -0.0007991, 0.0008441
2: -0.0108669, -0.0053976, -0.0109673, -0.0050852, -0.0044732, 0.0042350
3: 0.0011836, 0.0036730, 0.0010414, 0.0037187, -0.0019276, 0.0020360
4: -0.0015754, -0.0005168, -0.0015948, -0.0004563, -0.0008658, 0.0008197
5: -0.0147082, -0.0078293, -0.0148345, -0.0074363, -0.0056261, 0.0053265
6: 0.0035280, 0.0052739, 0.0034283, 0.0053060, -0.0013519, 0.0014280
7: 0.0059904, 0.0105077, 0.0057323, 0.0105906, -0.0034978, 0.0036946
8: 0.0035861, 0.0059617, 0.0034504, 0.0060053, -0.0018395, 0.0019430
9: -0.0087768, -0.0060222, -0.0088273, -0.0058648, -0.0022530, 0.0021330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014932, upper bound: 0.0014926
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016203, upper bound: 0.0015293
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 1.0016201, 1.0056851, 1.0014869, 1.0059855, -0.0034239, 0.0033364
1: -0.0008603, 0.0001526, -0.0008935, 0.0002275, -0.0008532, 0.0008314
2: -0.0108627, -0.0054950, -0.0112595, -0.0053190, -0.0044057, 0.0045213
3: 0.0012279, 0.0036711, 0.0011479, 0.0038517, -0.0020579, 0.0020053
4: -0.0015746, -0.0005357, -0.0016514, -0.0005016, -0.0008527, 0.0008751
5: -0.0147029, -0.0079518, -0.0152021, -0.0077305, -0.0055412, 0.0056866
6: 0.0035591, 0.0052726, 0.0035029, 0.0053993, -0.0014433, 0.0014064
7: 0.0060708, 0.0105042, 0.0059255, 0.0108320, -0.0037343, 0.0036389
8: 0.0036284, 0.0059599, 0.0035520, 0.0061323, -0.0019638, 0.0019136
9: -0.0087746, -0.0060712, -0.0089745, -0.0059826, -0.0022190, 0.0022771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015207, upper bound: 0.0014076
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015194
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 1.0015520, 1.0056291, 1.0010141, 1.0054914, -0.0029657, 0.0036182
1: -0.0008772, 0.0001386, -0.0010113, 0.0001044, -0.0007390, 0.0009016
2: -0.0107887, -0.0054051, -0.0106071, -0.0046948, -0.0047778, 0.0039161
3: 0.0011870, 0.0036374, 0.0008637, 0.0035548, -0.0017824, 0.0021747
4: -0.0015602, -0.0005183, -0.0015251, -0.0003808, -0.0009247, 0.0007580
5: -0.0146099, -0.0078388, -0.0143814, -0.0069453, -0.0060092, 0.0049254
6: 0.0035304, 0.0052490, 0.0033036, 0.0051910, -0.0012501, 0.0015252
7: 0.0059966, 0.0104431, 0.0054099, 0.0102931, -0.0032345, 0.0039462
8: 0.0035894, 0.0059278, 0.0032809, 0.0058489, -0.0017010, 0.0020753
9: -0.0087374, -0.0060259, -0.0086459, -0.0056682, -0.0024064, 0.0019724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014932, upper bound: 0.0015178
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014370, upper bound: 0.0015020
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0011091, 1.0056273, -0.0030886, 0.0036484
1: -0.0008787, 0.0001534, -0.0009876, 0.0001382, -0.0007696, 0.0009091
2: -0.0108669, -0.0053976, -0.0107864, -0.0048204, -0.0048177, 0.0040785
3: 0.0011836, 0.0036730, 0.0009209, 0.0036364, -0.0018563, 0.0021928
4: -0.0015754, -0.0005168, -0.0015598, -0.0004051, -0.0009324, 0.0007894
5: -0.0147082, -0.0078293, -0.0146071, -0.0071033, -0.0060593, 0.0051296
6: 0.0035280, 0.0052739, 0.0033437, 0.0052483, -0.0013020, 0.0015379
7: 0.0059904, 0.0105077, 0.0055136, 0.0104412, -0.0033686, 0.0039791
8: 0.0035861, 0.0059617, 0.0033354, 0.0059268, -0.0017715, 0.0020926
9: -0.0087768, -0.0060222, -0.0087363, -0.0057314, -0.0024264, 0.0020541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016203, upper bound: 0.0015495
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015442
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0012932, 1.0054303, 1.0012436, 1.0057005, -0.0034205, 0.0032552
1: -0.0009417, 0.0000892, -0.0009541, 0.0001565, -0.0008523, 0.0008111
2: -0.0105265, -0.0050633, -0.0108832, -0.0049979, -0.0042985, 0.0045167
3: 0.0010315, 0.0035181, 0.0010017, 0.0036804, -0.0020558, 0.0019565
4: -0.0015095, -0.0004521, -0.0015785, -0.0004394, -0.0008320, 0.0008742
5: -0.0142801, -0.0074088, -0.0147288, -0.0073265, -0.0054064, 0.0056809
6: 0.0034213, 0.0051653, 0.0034004, 0.0052792, -0.0014419, 0.0013722
7: 0.0057142, 0.0102265, 0.0056602, 0.0105212, -0.0037305, 0.0035503
8: 0.0034409, 0.0058139, 0.0034125, 0.0059688, -0.0019619, 0.0018671
9: -0.0086053, -0.0058538, -0.0087850, -0.0058208, -0.0021650, 0.0022749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015893, upper bound: 0.0014484
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015672, upper bound: 0.0014084
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0014200, 1.0055397, 1.0012380, 1.0057670, -0.0034042, 0.0033357
1: -0.0009101, 0.0001164, -0.0009555, 0.0001730, -0.0008482, 0.0008312
2: -0.0106708, -0.0052308, -0.0109709, -0.0049904, -0.0044048, 0.0044952
3: 0.0011077, 0.0035838, 0.0009983, 0.0037204, -0.0020460, 0.0020049
4: -0.0015374, -0.0004845, -0.0015955, -0.0004380, -0.0008525, 0.0008700
5: -0.0144616, -0.0076196, -0.0148391, -0.0073172, -0.0055400, 0.0056538
6: 0.0034748, 0.0052113, 0.0033980, 0.0053072, -0.0014350, 0.0014061
7: 0.0058526, 0.0103457, 0.0056541, 0.0105936, -0.0037128, 0.0036381
8: 0.0035137, 0.0058766, 0.0034093, 0.0060069, -0.0019525, 0.0019132
9: -0.0086780, -0.0059382, -0.0088292, -0.0058171, -0.0022185, 0.0022640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016475, upper bound: 0.0015293
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016312, upper bound: 0.0015194
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 1.0014273, 1.0055231, 1.0010141, 1.0054914, -0.0030501, 0.0034423
1: -0.0009083, 0.0001122, -0.0010113, 0.0001044, -0.0007600, 0.0008577
2: -0.0106489, -0.0052404, -0.0106071, -0.0046948, -0.0045455, 0.0040277
3: 0.0011121, 0.0035738, 0.0008637, 0.0035548, -0.0018332, 0.0020689
4: -0.0015332, -0.0004864, -0.0015251, -0.0003808, -0.0008798, 0.0007795
5: -0.0144340, -0.0076316, -0.0143814, -0.0069453, -0.0057171, 0.0050658
6: 0.0034778, 0.0052043, 0.0033036, 0.0051910, -0.0012857, 0.0014511
7: 0.0058605, 0.0103276, 0.0054099, 0.0102931, -0.0033266, 0.0037543
8: 0.0035178, 0.0058670, 0.0032809, 0.0058489, -0.0017494, 0.0019744
9: -0.0086670, -0.0059430, -0.0086459, -0.0056682, -0.0022894, 0.0020286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015373
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015309
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 1.0014151, 1.0055836, 1.0011091, 1.0056273, -0.0031651, 0.0034612
1: -0.0009113, 0.0001273, -0.0009876, 0.0001382, -0.0007887, 0.0008624
2: -0.0107289, -0.0052244, -0.0107864, -0.0048204, -0.0045705, 0.0041795
3: 0.0011048, 0.0036102, 0.0009209, 0.0036364, -0.0019023, 0.0020803
4: -0.0015487, -0.0004833, -0.0015598, -0.0004051, -0.0008846, 0.0008089
5: -0.0145347, -0.0076115, -0.0146071, -0.0071033, -0.0057484, 0.0052567
6: 0.0034727, 0.0052299, 0.0033437, 0.0052483, -0.0013342, 0.0014590
7: 0.0058473, 0.0103937, 0.0055136, 0.0104412, -0.0034520, 0.0037749
8: 0.0035109, 0.0059018, 0.0033354, 0.0059268, -0.0018154, 0.0019852
9: -0.0087073, -0.0059349, -0.0087363, -0.0057314, -0.0023019, 0.0021050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016664, upper bound: 0.0015495
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016605, upper bound: 0.0015499
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014932
time: 1.34 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014076, upper bound: 0.0015207
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0015963
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015178, upper bound: 0.0014932
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0014370
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015495, upper bound: 0.0016203
time: 1.18 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015442, upper bound: 0.0015963
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

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

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0015723
time: 1.16 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014067, upper bound: 0.0015189
time: 1.22 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0015963
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

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

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015178, upper bound: 0.0014928
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0014333
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015495, upper bound: 0.0016203
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015442, upper bound: 0.0015963
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 1.0012436, 1.0057005, 1.0012932, 1.0054303, -0.0032552, 0.0034205
1: -0.0009541, 0.0001565, -0.0009417, 0.0000892, -0.0008111, 0.0008523
2: -0.0108832, -0.0049979, -0.0105265, -0.0050633, -0.0045167, 0.0042985
3: 0.0010017, 0.0036804, 0.0010315, 0.0035181, -0.0019565, 0.0020558
4: -0.0015785, -0.0004394, -0.0015095, -0.0004521, -0.0008742, 0.0008320
5: -0.0147288, -0.0073265, -0.0142801, -0.0074088, -0.0056809, 0.0054064
6: 0.0034004, 0.0052792, 0.0034213, 0.0051653, -0.0013722, 0.0014419
7: 0.0056602, 0.0105212, 0.0057142, 0.0102265, -0.0035503, 0.0037305
8: 0.0034125, 0.0059688, 0.0034409, 0.0058139, -0.0018671, 0.0019619
9: -0.0087850, -0.0058208, -0.0086053, -0.0058538, -0.0022749, 0.0021650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014484, upper bound: 0.0015893
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0015672
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0014200, 1.0055397, -0.0033357, 0.0034042
1: -0.0009555, 0.0001730, -0.0009101, 0.0001164, -0.0008312, 0.0008482
2: -0.0109709, -0.0049904, -0.0106708, -0.0052308, -0.0044952, 0.0044048
3: 0.0009983, 0.0037204, 0.0011077, 0.0035838, -0.0020049, 0.0020460
4: -0.0015955, -0.0004380, -0.0015374, -0.0004845, -0.0008700, 0.0008525
5: -0.0148391, -0.0073172, -0.0144616, -0.0076196, -0.0056538, 0.0055400
6: 0.0033980, 0.0053072, 0.0034748, 0.0052113, -0.0014061, 0.0014350
7: 0.0056541, 0.0105936, 0.0058526, 0.0103457, -0.0036381, 0.0037128
8: 0.0034093, 0.0060069, 0.0035137, 0.0058766, -0.0019132, 0.0019525
9: -0.0088292, -0.0058171, -0.0086780, -0.0059382, -0.0022640, 0.0022185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016475
time: 1.14 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0016312
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015492, upper bound: 0.0015751
time: 1.31 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015430, upper bound: 0.0015532
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015623, upper bound: 0.0016495
time: 1.19 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016445
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0012436, 1.0057005, 1.0010141, 1.0054914, -0.0029681, 0.0033953
1: -0.0009541, 0.0001565, -0.0010113, 0.0001044, -0.0007396, 0.0008460
2: -0.0108832, -0.0049979, -0.0106071, -0.0046948, -0.0044835, 0.0039194
3: 0.0010017, 0.0036804, 0.0008637, 0.0035548, -0.0017839, 0.0020407
4: -0.0015785, -0.0004394, -0.0015251, -0.0003808, -0.0008678, 0.0007586
5: -0.0147288, -0.0073265, -0.0143814, -0.0069453, -0.0056390, 0.0049296
6: 0.0034004, 0.0052792, 0.0033036, 0.0051910, -0.0012512, 0.0014312
7: 0.0056602, 0.0105212, 0.0054099, 0.0102931, -0.0032372, 0.0037031
8: 0.0034125, 0.0059688, 0.0032809, 0.0058489, -0.0017024, 0.0019474
9: -0.0087850, -0.0058208, -0.0086459, -0.0056682, -0.0022581, 0.0019740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014484, upper bound: 0.0015893
time: 1.14 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014078, upper bound: 0.0015668
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0011091, 1.0056273, -0.0031045, 0.0033699
1: -0.0009555, 0.0001730, -0.0009876, 0.0001382, -0.0007735, 0.0008397
2: -0.0109709, -0.0049904, -0.0107864, -0.0048204, -0.0044499, 0.0040994
3: 0.0009983, 0.0037204, 0.0009209, 0.0036364, -0.0018659, 0.0020254
4: -0.0015955, -0.0004380, -0.0015598, -0.0004051, -0.0008613, 0.0007934
5: -0.0148391, -0.0073172, -0.0146071, -0.0071033, -0.0055968, 0.0051560
6: 0.0033980, 0.0053072, 0.0033437, 0.0052483, -0.0013086, 0.0014205
7: 0.0056541, 0.0105936, 0.0055136, 0.0104412, -0.0033858, 0.0036753
8: 0.0034093, 0.0060069, 0.0033354, 0.0059268, -0.0017806, 0.0019328
9: -0.0088292, -0.0058171, -0.0087363, -0.0057314, -0.0022412, 0.0020647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015312, upper bound: 0.0016423
time: 1.13 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0016312
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015492, upper bound: 0.0015751
time: 1.13 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015430, upper bound: 0.0015531
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015623, upper bound: 0.0016495
time: 1.20 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016445
time: 1.15 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.63 seconds
IS_A1_B1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014916, upper bound: 0.0015576
IS_A1_B1_A1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014594, upper bound: 0.0015474
IS_A1_B1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015587, upper bound: 0.0015797
IS_A1_B1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015505, upper bound: 0.0015761
IS_A1_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015576, upper bound: 0.0014916
IS_A1_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015474, upper bound: 0.0014595
IS_A1_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015797, upper bound: 0.0015587
IS_A1_B1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015761, upper bound: 0.0015505
IS_A1_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015843, upper bound: 0.0015475
IS_A1_B1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015842, upper bound: 0.0015420
IS_A1_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015915, upper bound: 0.0015781
IS_A1_B1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015943, upper bound: 0.0015779
IS_A1_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014932, upper bound: 0.0014926
IS_A1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0016203, upper bound: 0.0015293
IS_A1_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015207, upper bound: 0.0014076
IS_A1_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015194
IS_A1_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014932, upper bound: 0.0015178
IS_A1_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014370, upper bound: 0.0015020
IS_A1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0016203, upper bound: 0.0015495
IS_A1_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015442
IS_A1_B2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015893, upper bound: 0.0014484
IS_A1_B2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015672, upper bound: 0.0014084
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0016475, upper bound: 0.0015293
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0016312, upper bound: 0.0015194
IS_A1_B2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015373
IS_A1_B2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015309
IS_A1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0016664, upper bound: 0.0015495
IS_A1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0016605, upper bound: 0.0015499
IS_A2_B1_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014932
IS_A2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014076, upper bound: 0.0015207
IS_A2_B1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0015963
IS_A2_B1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015178, upper bound: 0.0014932
IS_A2_B1_B1_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0014370
IS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015495, upper bound: 0.0016203
IS_A2_B1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015442, upper bound: 0.0015963
IS_A2_B1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0015723
IS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014067, upper bound: 0.0015189
IS_A2_B1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0015963
IS_A2_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015178, upper bound: 0.0014928
IS_A2_B1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0014333
IS_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015495, upper bound: 0.0016203
IS_A2_B1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015442, upper bound: 0.0015963
IS_A2_B2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014484, upper bound: 0.0015893
IS_A2_B2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014084, upper bound: 0.0015672
IS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016475
IS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0016312
IS_A2_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015492, upper bound: 0.0015751
IS_A2_B2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015430, upper bound: 0.0015532
IS_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015623, upper bound: 0.0016495
IS_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016445
IS_A2_B2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014484, upper bound: 0.0015893
IS_A2_B2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0014078, upper bound: 0.0015668
IS_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015312, upper bound: 0.0016423
IS_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015194, upper bound: 0.0016312
IS_A2_B2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015492, upper bound: 0.0015751
IS_A2_B2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015430, upper bound: 0.0015531
IS_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015623, upper bound: 0.0016495
IS_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -0.0015634, upper bound: 0.0016445

## BFS IS instance: IS_A1_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0013121, 1.0057251, -0.0031586, 0.0033836
1: -0.0008787, 0.0001534, -0.0009370, 0.0001626, -0.0007870, 0.0008431
2: -0.0108669, -0.0053976, -0.0109158, -0.0050884, -0.0044680, 0.0041709
3: 0.0011836, 0.0036730, 0.0010429, 0.0036953, -0.0018984, 0.0020336
4: -0.0015754, -0.0005168, -0.0015848, -0.0004570, -0.0008648, 0.0008073
5: -0.0147082, -0.0078293, -0.0147698, -0.0074404, -0.0056195, 0.0052459
6: 0.0035280, 0.0052739, 0.0034293, 0.0052896, -0.0013315, 0.0014263
7: 0.0059904, 0.0105077, 0.0057350, 0.0105481, -0.0034449, 0.0036902
8: 0.0035861, 0.0059617, 0.0034518, 0.0059830, -0.0018117, 0.0019407
9: -0.0087768, -0.0060222, -0.0088014, -0.0058664, -0.0022503, 0.0021007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016203, upper bound: 0.0015293
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016203, upper bound: 0.0015293
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 1.0015463, 1.0056882, 1.0011817, 1.0056244, -0.0030866, 0.0035683
1: -0.0008787, 0.0001534, -0.0009695, 0.0001375, -0.0007691, 0.0008891
2: -0.0108669, -0.0053976, -0.0107827, -0.0049161, -0.0047120, 0.0040758
3: 0.0011836, 0.0036730, 0.0009645, 0.0036347, -0.0018551, 0.0021447
4: -0.0015754, -0.0005168, -0.0015591, -0.0004236, -0.0009120, 0.0007889
5: -0.0147082, -0.0078293, -0.0146023, -0.0072237, -0.0059264, 0.0051263
6: 0.0035280, 0.0052739, 0.0033743, 0.0052471, -0.0013011, 0.0015042
7: 0.0059904, 0.0105077, 0.0055927, 0.0104381, -0.0033663, 0.0038918
8: 0.0035861, 0.0059617, 0.0033770, 0.0059252, -0.0017703, 0.0020467
9: -0.0087768, -0.0060222, -0.0087344, -0.0057796, -0.0023732, 0.0020528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015433
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015433
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 1.0014200, 1.0055397, 1.0013096, 1.0057642, -0.0034021, 0.0032576
1: -0.0009101, 0.0001164, -0.0009376, 0.0001723, -0.0008477, 0.0008117
2: -0.0106708, -0.0052308, -0.0109673, -0.0050852, -0.0043016, 0.0044925
3: 0.0011077, 0.0035838, 0.0010414, 0.0037187, -0.0020448, 0.0019579
4: -0.0015374, -0.0004845, -0.0015948, -0.0004563, -0.0008326, 0.0008695
5: -0.0144616, -0.0076196, -0.0148345, -0.0074363, -0.0054103, 0.0056504
6: 0.0034748, 0.0052113, 0.0034283, 0.0053060, -0.0014341, 0.0013732
7: 0.0058526, 0.0103457, 0.0057323, 0.0105906, -0.0037105, 0.0035528
8: 0.0035137, 0.0058766, 0.0034504, 0.0060053, -0.0019513, 0.0018684
9: -0.0086780, -0.0059382, -0.0088273, -0.0058648, -0.0021665, 0.0022627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016312, upper bound: 0.0015194
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016312, upper bound: 0.0015194
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 1.0014930, 1.0055364, 1.0014869, 1.0059855, -0.0036213, 0.0032104
1: -0.0008920, 0.0001156, -0.0008935, 0.0002275, -0.0009023, 0.0008000
2: -0.0106664, -0.0053271, -0.0112595, -0.0053190, -0.0042393, 0.0047818
3: 0.0011515, 0.0035818, 0.0011479, 0.0038517, -0.0021765, 0.0019296
4: -0.0015366, -0.0005032, -0.0016514, -0.0005016, -0.0008205, 0.0009255
5: -0.0144561, -0.0077406, -0.0152021, -0.0077305, -0.0053320, 0.0060143
6: 0.0035055, 0.0052099, 0.0035029, 0.0053993, -0.0015265, 0.0013533
7: 0.0059321, 0.0103421, 0.0059255, 0.0108320, -0.0039495, 0.0035014
8: 0.0035555, 0.0058747, 0.0035520, 0.0061323, -0.0020770, 0.0018414
9: -0.0086758, -0.0059866, -0.0089745, -0.0059826, -0.0021352, 0.0024084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014768, upper bound: 0.0014655
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014768, upper bound: 0.0015194
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 1.0014151, 1.0055836, 1.0011817, 1.0056244, -0.0031632, 0.0033798
1: -0.0009113, 0.0001273, -0.0009695, 0.0001375, -0.0007882, 0.0008421
2: -0.0107289, -0.0052244, -0.0107827, -0.0049161, -0.0044629, 0.0041770
3: 0.0011048, 0.0036102, 0.0009645, 0.0036347, -0.0019012, 0.0020313
4: -0.0015487, -0.0004833, -0.0015591, -0.0004236, -0.0008638, 0.0008084
5: -0.0145347, -0.0076115, -0.0146023, -0.0072237, -0.0056132, 0.0052535
6: 0.0034727, 0.0052299, 0.0033743, 0.0052471, -0.0013334, 0.0014247
7: 0.0058473, 0.0103937, 0.0055927, 0.0104381, -0.0034499, 0.0036861
8: 0.0035109, 0.0059018, 0.0033770, 0.0059252, -0.0018143, 0.0019385
9: -0.0087073, -0.0059349, -0.0087344, -0.0057796, -0.0022478, 0.0021037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016591, upper bound: 0.0015461
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016591, upper bound: 0.0015461
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 1.0014881, 1.0055804, 1.0013490, 1.0058484, -0.0033830, 0.0033235
1: -0.0008932, 0.0001265, -0.0009278, 0.0001933, -0.0008430, 0.0008281
2: -0.0107246, -0.0053207, -0.0110784, -0.0051370, -0.0043887, 0.0044672
3: 0.0011486, 0.0036083, 0.0010650, 0.0037693, -0.0020333, 0.0019975
4: -0.0015478, -0.0005019, -0.0016163, -0.0004664, -0.0008494, 0.0008646
5: -0.0145293, -0.0077326, -0.0149742, -0.0075016, -0.0055198, 0.0056186
6: 0.0035035, 0.0052285, 0.0034448, 0.0053415, -0.0014261, 0.0014010
7: 0.0059269, 0.0103901, 0.0057752, 0.0106823, -0.0036896, 0.0036248
8: 0.0035527, 0.0058999, 0.0034729, 0.0060536, -0.0019403, 0.0019062
9: -0.0087051, -0.0059834, -0.0088833, -0.0058909, -0.0022104, 0.0022499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015031
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015499
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0015463, 1.0056882, -0.0033836, 0.0031586
1: -0.0009370, 0.0001626, -0.0008787, 0.0001534, -0.0008431, 0.0007870
2: -0.0109158, -0.0050884, -0.0108669, -0.0053976, -0.0041709, 0.0044680
3: 0.0010429, 0.0036953, 0.0011836, 0.0036730, -0.0020336, 0.0018984
4: -0.0015848, -0.0004570, -0.0015754, -0.0005168, -0.0008073, 0.0008648
5: -0.0147698, -0.0074404, -0.0147082, -0.0078293, -0.0052459, 0.0056195
6: 0.0034293, 0.0052896, 0.0035280, 0.0052739, -0.0014263, 0.0013315
7: 0.0057350, 0.0105481, 0.0059904, 0.0105077, -0.0036902, 0.0034449
8: 0.0034518, 0.0059830, 0.0035861, 0.0059617, -0.0019407, 0.0018117
9: -0.0088014, -0.0058664, -0.0087768, -0.0060222, -0.0021007, 0.0022503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0015463, 1.0056882, -0.0035683, 0.0030866
1: -0.0009695, 0.0001375, -0.0008787, 0.0001534, -0.0008891, 0.0007691
2: -0.0107827, -0.0049161, -0.0108669, -0.0053976, -0.0040758, 0.0047120
3: 0.0009645, 0.0036347, 0.0011836, 0.0036730, -0.0021447, 0.0018551
4: -0.0015591, -0.0004236, -0.0015754, -0.0005168, -0.0007889, 0.0009120
5: -0.0146023, -0.0072237, -0.0147082, -0.0078293, -0.0051263, 0.0059264
6: 0.0033743, 0.0052471, 0.0035280, 0.0052739, -0.0015042, 0.0013011
7: 0.0055927, 0.0104381, 0.0059904, 0.0105077, -0.0038918, 0.0033663
8: 0.0033770, 0.0059252, 0.0035861, 0.0059617, -0.0020467, 0.0017703
9: -0.0087344, -0.0057796, -0.0087768, -0.0060222, -0.0020528, 0.0023732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015432, upper bound: 0.0015963
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015432, upper bound: 0.0015963
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0012405, 1.0057279, -0.0030885, 0.0031725
1: -0.0009376, 0.0001723, -0.0009549, 0.0001633, -0.0007696, 0.0007905
2: -0.0109673, -0.0050852, -0.0109194, -0.0049937, -0.0041892, 0.0040783
3: 0.0010414, 0.0037187, 0.0009998, 0.0036969, -0.0018563, 0.0019067
4: -0.0015948, -0.0004563, -0.0015855, -0.0004386, -0.0008108, 0.0007893
5: -0.0148345, -0.0074363, -0.0147743, -0.0073213, -0.0052689, 0.0051294
6: 0.0034283, 0.0053060, 0.0033990, 0.0052907, -0.0013019, 0.0013373
7: 0.0057323, 0.0105906, 0.0056567, 0.0105511, -0.0033684, 0.0034600
8: 0.0034504, 0.0060053, 0.0034107, 0.0059845, -0.0017714, 0.0018196
9: -0.0088273, -0.0058648, -0.0088032, -0.0058187, -0.0021099, 0.0020540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.19 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0012380, 1.0057670, -0.0032879, 0.0031023
1: -0.0009695, 0.0001375, -0.0009555, 0.0001730, -0.0008192, 0.0007730
2: -0.0107827, -0.0049161, -0.0109709, -0.0049904, -0.0040966, 0.0043416
3: 0.0009645, 0.0036347, 0.0009983, 0.0037204, -0.0019761, 0.0018646
4: -0.0015591, -0.0004236, -0.0015955, -0.0004380, -0.0007929, 0.0008403
5: -0.0146023, -0.0072237, -0.0148391, -0.0073172, -0.0051524, 0.0054606
6: 0.0033743, 0.0052471, 0.0033980, 0.0053072, -0.0013860, 0.0013077
7: 0.0055927, 0.0104381, 0.0056541, 0.0105936, -0.0035859, 0.0033835
8: 0.0033770, 0.0059252, 0.0034093, 0.0060069, -0.0018858, 0.0017794
9: -0.0087344, -0.0057796, -0.0088292, -0.0058171, -0.0020633, 0.0021867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015433, upper bound: 0.0015963
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015433, upper bound: 0.0015963
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0014200, 1.0055397, -0.0032576, 0.0034021
1: -0.0009376, 0.0001723, -0.0009101, 0.0001164, -0.0008117, 0.0008477
2: -0.0109673, -0.0050852, -0.0106708, -0.0052308, -0.0044925, 0.0043016
3: 0.0010414, 0.0037187, 0.0011077, 0.0035838, -0.0019579, 0.0020448
4: -0.0015948, -0.0004563, -0.0015374, -0.0004845, -0.0008695, 0.0008326
5: -0.0148345, -0.0074363, -0.0144616, -0.0076196, -0.0056504, 0.0054103
6: 0.0034283, 0.0053060, 0.0034748, 0.0052113, -0.0013732, 0.0014341
7: 0.0057323, 0.0105906, 0.0058526, 0.0103457, -0.0035528, 0.0037105
8: 0.0034504, 0.0060053, 0.0035137, 0.0058766, -0.0018684, 0.0019513
9: -0.0088273, -0.0058648, -0.0086780, -0.0059382, -0.0022627, 0.0021665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015193, upper bound: 0.0016312
time: 1.14 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015193, upper bound: 0.0016312
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A2

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0014768
time: 1.30 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0014151, 1.0055836, -0.0033798, 0.0031632
1: -0.0009695, 0.0001375, -0.0009113, 0.0001273, -0.0008421, 0.0007882
2: -0.0107827, -0.0049161, -0.0107289, -0.0052244, -0.0041770, 0.0044629
3: 0.0009645, 0.0036347, 0.0011048, 0.0036102, -0.0020313, 0.0019012
4: -0.0015591, -0.0004236, -0.0015487, -0.0004833, -0.0008084, 0.0008638
5: -0.0146023, -0.0072237, -0.0145347, -0.0076115, -0.0052535, 0.0056132
6: 0.0033743, 0.0052471, 0.0034727, 0.0052299, -0.0014247, 0.0013334
7: 0.0055927, 0.0104381, 0.0058473, 0.0103937, -0.0036861, 0.0034499
8: 0.0033770, 0.0059252, 0.0035109, 0.0059018, -0.0019385, 0.0018143
9: -0.0087344, -0.0057796, -0.0087073, -0.0059349, -0.0021037, 0.0022478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015588, upper bound: 0.0016425
time: 1.32 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015588, upper bound: 0.0016445
time: 1.31 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0014881, 1.0055804, -0.0033235, 0.0033830
1: -0.0009278, 0.0001933, -0.0008932, 0.0001265, -0.0008281, 0.0008430
2: -0.0110784, -0.0051370, -0.0107246, -0.0053207, -0.0044672, 0.0043887
3: 0.0010650, 0.0037693, 0.0011486, 0.0036083, -0.0019975, 0.0020333
4: -0.0016163, -0.0004664, -0.0015478, -0.0005019, -0.0008646, 0.0008494
5: -0.0149742, -0.0075016, -0.0145293, -0.0077326, -0.0056186, 0.0055198
6: 0.0034448, 0.0053415, 0.0035035, 0.0052285, -0.0014010, 0.0014261
7: 0.0057752, 0.0106823, 0.0059269, 0.0103901, -0.0036248, 0.0036896
8: 0.0034729, 0.0060536, 0.0035527, 0.0058999, -0.0019062, 0.0019403
9: -0.0088833, -0.0058909, -0.0087051, -0.0059834, -0.0022499, 0.0022104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
time: 1.17 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016445
time: 1.71 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 1.0012380, 1.0057670, 1.0011817, 1.0056244, -0.0031023, 0.0032879
1: -0.0009555, 0.0001730, -0.0009695, 0.0001375, -0.0007730, 0.0008192
2: -0.0109709, -0.0049904, -0.0107827, -0.0049161, -0.0043416, 0.0040966
3: 0.0009983, 0.0037204, 0.0009645, 0.0036347, -0.0018646, 0.0019761
4: -0.0015955, -0.0004380, -0.0015591, -0.0004236, -0.0008403, 0.0007929
5: -0.0148391, -0.0073172, -0.0146023, -0.0072237, -0.0054606, 0.0051524
6: 0.0033980, 0.0053072, 0.0033743, 0.0052471, -0.0013077, 0.0013860
7: 0.0056541, 0.0105936, 0.0055927, 0.0104381, -0.0033835, 0.0035859
8: 0.0034093, 0.0060069, 0.0033770, 0.0059252, -0.0017794, 0.0018858
9: -0.0088292, -0.0058171, -0.0087344, -0.0057796, -0.0021867, 0.0020633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015193, upper bound: 0.0016312
time: 1.17 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015193, upper bound: 0.0016312
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 1.0013114, 1.0057641, 1.0013490, 1.0058484, -0.0033254, 0.0032434
1: -0.0009372, 0.0001723, -0.0009278, 0.0001933, -0.0008286, 0.0008082
2: -0.0109671, -0.0050875, -0.0110784, -0.0051370, -0.0042828, 0.0043911
3: 0.0010425, 0.0037186, 0.0010650, 0.0037693, -0.0019986, 0.0019494
4: -0.0015948, -0.0004568, -0.0016163, -0.0004664, -0.0008289, 0.0008499
5: -0.0148343, -0.0074393, -0.0149742, -0.0075016, -0.0053867, 0.0055229
6: 0.0034290, 0.0053059, 0.0034448, 0.0053415, -0.0014018, 0.0013672
7: 0.0057342, 0.0105904, 0.0057752, 0.0106823, -0.0036268, 0.0035373
8: 0.0034514, 0.0060053, 0.0034729, 0.0060536, -0.0019073, 0.0018603
9: -0.0088272, -0.0058660, -0.0088833, -0.0058909, -0.0021571, 0.0022116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0014719
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0011046, 1.0056679, -0.0031018, 0.0031625
1: -0.0009695, 0.0001375, -0.0009887, 0.0001483, -0.0007729, 0.0007880
2: -0.0107827, -0.0049161, -0.0108400, -0.0048142, -0.0041761, 0.0040959
3: 0.0009645, 0.0036347, 0.0009181, 0.0036608, -0.0018643, 0.0019008
4: -0.0015591, -0.0004236, -0.0015702, -0.0004039, -0.0008083, 0.0007928
5: -0.0146023, -0.0072237, -0.0146744, -0.0070956, -0.0052524, 0.0051516
6: 0.0033743, 0.0052471, 0.0033418, 0.0052654, -0.0013075, 0.0013331
7: 0.0055927, 0.0104381, 0.0055086, 0.0104855, -0.0033830, 0.0034492
8: 0.0033770, 0.0059252, 0.0033328, 0.0059501, -0.0017791, 0.0018139
9: -0.0087344, -0.0057796, -0.0087632, -0.0057283, -0.0021033, 0.0020629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
time: 1.16 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0011774, 1.0056648, -0.0030520, 0.0033882
1: -0.0009278, 0.0001933, -0.0009706, 0.0001476, -0.0007605, 0.0008443
2: -0.0110784, -0.0051370, -0.0108360, -0.0049105, -0.0044741, 0.0040302
3: 0.0010650, 0.0037693, 0.0009619, 0.0036590, -0.0018344, 0.0020364
4: -0.0016163, -0.0004664, -0.0015694, -0.0004225, -0.0008660, 0.0007800
5: -0.0149742, -0.0075016, -0.0146694, -0.0072166, -0.0056273, 0.0050689
6: 0.0034448, 0.0053415, 0.0033725, 0.0052641, -0.0012865, 0.0014283
7: 0.0057752, 0.0106823, 0.0055880, 0.0104822, -0.0033287, 0.0036953
8: 0.0034729, 0.0060536, 0.0033745, 0.0059483, -0.0017505, 0.0019433
9: -0.0088833, -0.0058909, -0.0087612, -0.0057768, -0.0022534, 0.0020298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
time: 1.12 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016445
time: 1.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.76 seconds
IS_A1_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0016203, upper bound: 0.0015293
IS_A1_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0016203, upper bound: 0.0015293
IS_A1_B2_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015433
IS_A1_B2_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015963, upper bound: 0.0015433
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0016312, upper bound: 0.0015194
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0016312, upper bound: 0.0015194
IS_A1_B2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0014768, upper bound: 0.0014655
IS_A1_B2_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0014768, upper bound: 0.0015194
IS_A1_B2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0016591, upper bound: 0.0015461
IS_A1_B2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0016591, upper bound: 0.0015461
IS_A1_B2_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015031
IS_A1_B2_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015638, upper bound: 0.0015499
IS_A2_B1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B1_B1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015432, upper bound: 0.0015963
IS_A2_B1_B1_A2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015432, upper bound: 0.0015963
IS_A2_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016203
IS_A2_B1_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015433, upper bound: 0.0015963
IS_A2_B1_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015433, upper bound: 0.0015963
IS_A2_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015193, upper bound: 0.0016312
IS_A2_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015193, upper bound: 0.0016312
IS_A2_B2_B1_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0014768
IS_A2_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
IS_A2_B2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015588, upper bound: 0.0016425
IS_A2_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015588, upper bound: 0.0016445
IS_A2_B2_B1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
IS_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016445
IS_A2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015193, upper bound: 0.0016312
IS_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015193, upper bound: 0.0016312
IS_A2_B2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0014719
IS_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
IS_A2_B2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
IS_A2_B2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
IS_A2_B2_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016059
IS_A2_B2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -0.0015164, upper bound: 0.0016445

## BFS IS instance: IS_A1_B2_A1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 1.0016177, 1.0056852, 1.0013121, 1.0057251, -0.0030798, 0.0033814
1: -0.0008609, 0.0001526, -0.0009370, 0.0001626, -0.0007674, 0.0008426
2: -0.0108629, -0.0054918, -0.0109158, -0.0050884, -0.0044651, 0.0040669
3: 0.0012265, 0.0036712, 0.0010429, 0.0036953, -0.0018511, 0.0020323
4: -0.0015746, -0.0005350, -0.0015848, -0.0004570, -0.0008642, 0.0007871
5: -0.0147032, -0.0079478, -0.0147698, -0.0074404, -0.0056159, 0.0051150
6: 0.0035581, 0.0052727, 0.0034293, 0.0052896, -0.0012983, 0.0014254
7: 0.0060682, 0.0105044, 0.0057350, 0.0105481, -0.0033590, 0.0036879
8: 0.0036270, 0.0059600, 0.0034518, 0.0059830, -0.0017665, 0.0019394
9: -0.0087748, -0.0060696, -0.0088014, -0.0058664, -0.0022489, 0.0020483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014930, upper bound: 0.0014481
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014930, upper bound: 0.0015293
time: 1.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 1.0017924, 1.0059015, 1.0013121, 1.0057251, -0.0029869, 0.0036623
1: -0.0008173, 0.0002065, -0.0009370, 0.0001626, -0.0007443, 0.0009126
2: -0.0111484, -0.0057227, -0.0109158, -0.0050884, -0.0048361, 0.0039442
3: 0.0013316, 0.0038012, 0.0010429, 0.0036953, -0.0017952, 0.0022012
4: -0.0016299, -0.0005797, -0.0015848, -0.0004570, -0.0009360, 0.0007634
5: -0.0150624, -0.0082382, -0.0147698, -0.0074404, -0.0060825, 0.0049607
6: 0.0036318, 0.0053638, 0.0034293, 0.0052896, -0.0012591, 0.0015438
7: 0.0062589, 0.0107402, 0.0057350, 0.0105481, -0.0032576, 0.0039943
8: 0.0037273, 0.0060840, 0.0034518, 0.0059830, -0.0017132, 0.0021006
9: -0.0089186, -0.0061859, -0.0088014, -0.0058664, -0.0024357, 0.0019865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014930, upper bound: 0.0014481
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014930, upper bound: 0.0014460
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

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

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015290, upper bound: 0.0014926
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015290, upper bound: 0.0015293
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0016576, 1.0057553, 1.0013096, 1.0057642, -0.0032317, 0.0035360
1: -0.0008509, 0.0001701, -0.0009376, 0.0001723, -0.0008052, 0.0008811
2: -0.0109555, -0.0055445, -0.0109673, -0.0050852, -0.0046692, 0.0042674
3: 0.0012505, 0.0037134, 0.0010414, 0.0037187, -0.0019423, 0.0021252
4: -0.0015925, -0.0005452, -0.0015948, -0.0004563, -0.0009037, 0.0008259
5: -0.0148197, -0.0080141, -0.0148345, -0.0074363, -0.0058727, 0.0053673
6: 0.0035749, 0.0053022, 0.0034283, 0.0053060, -0.0013623, 0.0014905
7: 0.0061117, 0.0105809, 0.0057323, 0.0105906, -0.0035246, 0.0038565
8: 0.0036499, 0.0060002, 0.0034504, 0.0060053, -0.0018536, 0.0020281
9: -0.0088214, -0.0060961, -0.0088273, -0.0058648, -0.0023517, 0.0021493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015290, upper bound: 0.0014926
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015290, upper bound: 0.0015293
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0014859, 1.0055805, 1.0011817, 1.0056244, -0.0030842, 0.0033776
1: -0.0008937, 0.0001266, -0.0009695, 0.0001375, -0.0007685, 0.0008416
2: -0.0107247, -0.0053178, -0.0107827, -0.0049161, -0.0044601, 0.0040726
3: 0.0011473, 0.0036083, 0.0009645, 0.0036347, -0.0018537, 0.0020300
4: -0.0015479, -0.0005014, -0.0015591, -0.0004236, -0.0008632, 0.0007882
5: -0.0145294, -0.0077290, -0.0146023, -0.0072237, -0.0056096, 0.0051223
6: 0.0035025, 0.0052286, 0.0033743, 0.0052471, -0.0013001, 0.0014238
7: 0.0059245, 0.0103903, 0.0055927, 0.0104381, -0.0033637, 0.0036838
8: 0.0035515, 0.0059000, 0.0033770, 0.0059252, -0.0017690, 0.0019373
9: -0.0087052, -0.0059820, -0.0087344, -0.0057796, -0.0022463, 0.0020512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015166
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015166
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0016536, 1.0057963, 1.0011817, 1.0056244, -0.0029914, 0.0036615
1: -0.0008519, 0.0001803, -0.0009695, 0.0001375, -0.0007454, 0.0009124
2: -0.0110097, -0.0055392, -0.0107827, -0.0049161, -0.0048350, 0.0039501
3: 0.0012481, 0.0037380, 0.0009645, 0.0036347, -0.0017979, 0.0022007
4: -0.0016030, -0.0005442, -0.0015591, -0.0004236, -0.0009358, 0.0007645
5: -0.0148878, -0.0080074, -0.0146023, -0.0072237, -0.0060812, 0.0049682
6: 0.0035732, 0.0053195, 0.0033743, 0.0052471, -0.0012610, 0.0015435
7: 0.0061073, 0.0106256, 0.0055927, 0.0104381, -0.0032625, 0.0039934
8: 0.0036476, 0.0060237, 0.0033770, 0.0059252, -0.0017157, 0.0021001
9: -0.0088487, -0.0060935, -0.0087344, -0.0057796, -0.0024352, 0.0019895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015166
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015166
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2_B1

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015723
time: 1.16 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0016203
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2_B2

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015723
time: 1.16 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0016203
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0013121, 1.0057251, -0.0030866, 0.0030911
1: -0.0009376, 0.0001723, -0.0009370, 0.0001626, -0.0007691, 0.0007702
2: -0.0109673, -0.0050852, -0.0109158, -0.0050884, -0.0040818, 0.0040758
3: 0.0010414, 0.0037187, 0.0010429, 0.0036953, -0.0018551, 0.0018579
4: -0.0015948, -0.0004563, -0.0015848, -0.0004570, -0.0007900, 0.0007889
5: -0.0148345, -0.0074363, -0.0147698, -0.0074404, -0.0051339, 0.0051263
6: 0.0034283, 0.0053060, 0.0034293, 0.0052896, -0.0013011, 0.0013030
7: 0.0057323, 0.0105906, 0.0057350, 0.0105481, -0.0033663, 0.0033713
8: 0.0034504, 0.0060053, 0.0034518, 0.0059830, -0.0017703, 0.0017730
9: -0.0088273, -0.0058648, -0.0088014, -0.0058664, -0.0020558, 0.0020528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0014928
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0016203
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0014889, 1.0059466, -0.0033671, 0.0030004
1: -0.0009376, 0.0001723, -0.0008930, 0.0002178, -0.0008390, 0.0007476
2: -0.0109673, -0.0050852, -0.0112082, -0.0053218, -0.0039620, 0.0044462
3: 0.0010414, 0.0037187, 0.0011491, 0.0038284, -0.0020237, 0.0018033
4: -0.0015948, -0.0004563, -0.0016414, -0.0005021, -0.0007668, 0.0008606
5: -0.0148345, -0.0074363, -0.0151375, -0.0077339, -0.0049831, 0.0055922
6: 0.0034283, 0.0053060, 0.0035038, 0.0053829, -0.0014194, 0.0012648
7: 0.0057323, 0.0105906, 0.0059277, 0.0107895, -0.0036723, 0.0032723
8: 0.0034504, 0.0060053, 0.0035532, 0.0061100, -0.0019312, 0.0017209
9: -0.0088273, -0.0058648, -0.0089487, -0.0059840, -0.0019955, 0.0022394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0014928
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0014922
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0014908, 1.0055364, -0.0032554, 0.0033199
1: -0.0009376, 0.0001723, -0.0008925, 0.0001156, -0.0008112, 0.0008272
2: -0.0109673, -0.0050852, -0.0106665, -0.0053243, -0.0043839, 0.0042987
3: 0.0010414, 0.0037187, 0.0011503, 0.0035818, -0.0019566, 0.0019954
4: -0.0015948, -0.0004563, -0.0015366, -0.0005026, -0.0008485, 0.0008320
5: -0.0148345, -0.0074363, -0.0144563, -0.0077371, -0.0055138, 0.0054066
6: 0.0034283, 0.0053060, 0.0035046, 0.0052100, -0.0013723, 0.0013995
7: 0.0057323, 0.0105906, 0.0059298, 0.0103422, -0.0035504, 0.0036209
8: 0.0034504, 0.0060053, 0.0035543, 0.0058747, -0.0018671, 0.0019042
9: -0.0088273, -0.0058648, -0.0086759, -0.0059852, -0.0022080, 0.0021650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014466, upper bound: 0.0015290
time: 1.48 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014466, upper bound: 0.0016475
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0016576, 1.0057553, -0.0035360, 0.0032317
1: -0.0009376, 0.0001723, -0.0008509, 0.0001701, -0.0008811, 0.0008052
2: -0.0109673, -0.0050852, -0.0109555, -0.0055445, -0.0042674, 0.0046692
3: 0.0010414, 0.0037187, 0.0012505, 0.0037134, -0.0021252, 0.0019423
4: -0.0015948, -0.0004563, -0.0015925, -0.0005452, -0.0008259, 0.0009037
5: -0.0148345, -0.0074363, -0.0148197, -0.0080141, -0.0053673, 0.0058727
6: 0.0034283, 0.0053060, 0.0035749, 0.0053022, -0.0014905, 0.0013623
7: 0.0057323, 0.0105906, 0.0061117, 0.0105809, -0.0038565, 0.0035246
8: 0.0034504, 0.0060053, 0.0036499, 0.0060002, -0.0020281, 0.0018536
9: -0.0088273, -0.0058648, -0.0088214, -0.0060961, -0.0021493, 0.0023517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014466, upper bound: 0.0015290
time: 1.18 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014466, upper bound: 0.0016475
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A2_A2

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
time: 1.46 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1_B1

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
time: 1.22 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
time: 1.56 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
time: 1.20 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 1.0013490, 1.0058484, 1.0014930, 1.0055364, -0.0032609, 0.0033782
1: -0.0009278, 0.0001933, -0.0008920, 0.0001156, -0.0008125, 0.0008418
2: -0.0110784, -0.0051370, -0.0106664, -0.0053271, -0.0044609, 0.0043060
3: 0.0010650, 0.0037693, 0.0011515, 0.0035818, -0.0019599, 0.0020304
4: -0.0016163, -0.0004664, -0.0015366, -0.0005032, -0.0008634, 0.0008334
5: -0.0149742, -0.0075016, -0.0144561, -0.0077406, -0.0056107, 0.0054158
6: 0.0034448, 0.0053415, 0.0035055, 0.0052099, -0.0013746, 0.0014241
7: 0.0057752, 0.0106823, 0.0059321, 0.0103421, -0.0035565, 0.0036845
8: 0.0034729, 0.0060536, 0.0035555, 0.0058747, -0.0018703, 0.0019376
9: -0.0088833, -0.0058909, -0.0086758, -0.0059866, -0.0022468, 0.0021687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015165, upper bound: 0.0016425
time: 1.18 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015165, upper bound: 0.0016424
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 1.0013096, 1.0057642, 1.0011817, 1.0056244, -0.0030231, 0.0032858
1: -0.0009376, 0.0001723, -0.0009695, 0.0001375, -0.0007533, 0.0008187
2: -0.0109673, -0.0050852, -0.0107827, -0.0049161, -0.0043389, 0.0039919
3: 0.0010414, 0.0037187, 0.0009645, 0.0036347, -0.0018170, 0.0019749
4: -0.0015948, -0.0004563, -0.0015591, -0.0004236, -0.0008398, 0.0007726
5: -0.0148345, -0.0074363, -0.0146023, -0.0072237, -0.0054572, 0.0050208
6: 0.0034283, 0.0053060, 0.0033743, 0.0052471, -0.0012743, 0.0013851
7: 0.0057323, 0.0105906, 0.0055927, 0.0104381, -0.0032971, 0.0035837
8: 0.0034504, 0.0060053, 0.0033770, 0.0059252, -0.0017339, 0.0018846
9: -0.0088273, -0.0058648, -0.0087344, -0.0057796, -0.0021853, 0.0020105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0015122
time: 1.49 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0016423
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 1.0014869, 1.0059855, 1.0011817, 1.0056244, -0.0029319, 0.0035692
1: -0.0008935, 0.0002275, -0.0009695, 0.0001375, -0.0007306, 0.0008893
2: -0.0112595, -0.0053190, -0.0107827, -0.0049161, -0.0047130, 0.0038716
3: 0.0011479, 0.0038517, 0.0009645, 0.0036347, -0.0017622, 0.0021452
4: -0.0016514, -0.0005016, -0.0015591, -0.0004236, -0.0009122, 0.0007493
5: -0.0152021, -0.0077305, -0.0146023, -0.0072237, -0.0059277, 0.0048695
6: 0.0035029, 0.0053993, 0.0033743, 0.0052471, -0.0012359, 0.0015045
7: 0.0059255, 0.0108320, 0.0055927, 0.0104381, -0.0031977, 0.0038927
8: 0.0035520, 0.0061323, 0.0033770, 0.0059252, -0.0016816, 0.0020471
9: -0.0089745, -0.0059826, -0.0087344, -0.0057796, -0.0023737, 0.0019499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0015122
time: 1.26 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0016423
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 1.0013139, 1.0057251, 1.0013490, 1.0058484, -0.0033214, 0.0032178
1: -0.0009366, 0.0001626, -0.0009278, 0.0001933, -0.0008276, 0.0008018
2: -0.0109157, -0.0050907, -0.0110784, -0.0051370, -0.0042491, 0.0043859
3: 0.0010439, 0.0036952, 0.0010650, 0.0037693, -0.0019963, 0.0019340
4: -0.0015848, -0.0004574, -0.0016163, -0.0004664, -0.0008224, 0.0008489
5: -0.0147696, -0.0074433, -0.0149742, -0.0075016, -0.0053443, 0.0055163
6: 0.0034300, 0.0052895, 0.0034448, 0.0053415, -0.0014001, 0.0013564
7: 0.0057369, 0.0105479, 0.0057752, 0.0106823, -0.0036225, 0.0035095
8: 0.0034528, 0.0059829, 0.0034729, 0.0060536, -0.0019050, 0.0018456
9: -0.0088013, -0.0058676, -0.0088833, -0.0058909, -0.0021401, 0.0022090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
time: 1.11 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1_B1

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
time: 1.26 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1_B2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A2_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015165, upper bound: 0.0016425
time: 1.56 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015165, upper bound: 0.0016425
time: 1.10 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.91 seconds
IS_A1_B2_A1_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014930, upper bound: 0.0014481
IS_A1_B2_A1_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014930, upper bound: 0.0015293
IS_A1_B2_A1_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014930, upper bound: 0.0014481
IS_A1_B2_A1_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014930, upper bound: 0.0014460
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015290, upper bound: 0.0014926
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015290, upper bound: 0.0015293
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015290, upper bound: 0.0014926
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015290, upper bound: 0.0015293
IS_A1_B2_A2_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015166
IS_A1_B2_A2_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015166
IS_A1_B2_A2_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015166
IS_A1_B2_A2_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015883, upper bound: 0.0015166
IS_A2_B1_B1_A1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015723
IS_A2_B1_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0016203
IS_A2_B1_B1_A1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015723
IS_A2_B1_B1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0016203
IS_A2_B1_B2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0014928
IS_A2_B1_B2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0016203
IS_A2_B1_B2_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0014928
IS_A2_B1_B2_A1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014457, upper bound: 0.0014922
IS_A2_B2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014466, upper bound: 0.0015290
IS_A2_B2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014466, upper bound: 0.0016475
IS_A2_B2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014466, upper bound: 0.0015290
IS_A2_B2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014466, upper bound: 0.0016475
IS_A2_B2_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
IS_A2_B2_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014030, upper bound: 0.0016312
IS_A2_B2_B1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
IS_A2_B2_B1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
IS_A2_B2_B1_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
IS_A2_B2_B1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
IS_A2_B2_B1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015165, upper bound: 0.0016425
IS_A2_B2_B1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015165, upper bound: 0.0016424
IS_A2_B2_B2_A1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0015122
IS_A2_B2_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0016423
IS_A2_B2_B2_A1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0015122
IS_A2_B2_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014476, upper bound: 0.0016423
IS_A2_B2_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
IS_A2_B2_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0014024, upper bound: 0.0016312
IS_A2_B2_B2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
IS_A2_B2_B2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016253
IS_A2_B2_B2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
IS_A2_B2_B2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015293, upper bound: 0.0016495
IS_A2_B2_B2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015165, upper bound: 0.0016425
IS_A2_B2_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.91
Output dim: 0, lower bound: -0.0015165, upper bound: 0.0016425

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2_B1_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014714, upper bound: 0.0016206
time: 1.24 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0016230
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2_B2_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010288, upper bound: 0.0013104
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 21.09 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008675, upper bound: 0.0011030
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008009, upper bound: 0.0009020
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2_B1_A2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013737, upper bound: 0.0016376
time: 1.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013665, upper bound: 0.0016231
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1_B1_A2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013737, upper bound: 0.0016601
time: 1.07 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013666, upper bound: 0.0016494
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1_B2_A2

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009807, upper bound: 0.0009282
time: 0.90 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 20.71 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009710, upper bound: 0.0010194
time: 0.96 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008122, upper bound: 0.0009449
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A2_A2_B1

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
time: 1.17 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A2_A2_B2

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
time: 1.21 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0013630, 1.0054271, -0.0032339, 0.0031924
1: -0.0009695, 0.0001375, -0.0009243, 0.0000883, -0.0008058, 0.0007955
2: -0.0107827, -0.0049161, -0.0105222, -0.0051555, -0.0042156, 0.0042703
3: 0.0009645, 0.0036347, 0.0010735, 0.0035161, -0.0019437, 0.0019188
4: -0.0015591, -0.0004236, -0.0015087, -0.0004700, -0.0008159, 0.0008265
5: -0.0146023, -0.0072237, -0.0142747, -0.0075248, -0.0053021, 0.0053710
6: 0.0033743, 0.0052471, 0.0034507, 0.0051639, -0.0013632, 0.0013457
7: 0.0055927, 0.0104381, 0.0057904, 0.0102229, -0.0035270, 0.0034818
8: 0.0033770, 0.0059252, 0.0034810, 0.0058120, -0.0018548, 0.0018311
9: -0.0087344, -0.0057796, -0.0086032, -0.0059002, -0.0021232, 0.0021508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015224, upper bound: 0.0015924
time: 1.20 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014496, upper bound: 0.0015894
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1_B1_B2

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015224, upper bound: 0.0016341
time: 1.21 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014496, upper bound: 0.0016395
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0015507, 1.0056390, -0.0035164, 0.0030936
1: -0.0009695, 0.0001375, -0.0008776, 0.0001411, -0.0008762, 0.0007708
2: -0.0107827, -0.0049161, -0.0108018, -0.0054034, -0.0040851, 0.0046433
3: 0.0009645, 0.0036347, 0.0011863, 0.0036434, -0.0021134, 0.0018593
4: -0.0015591, -0.0004236, -0.0015628, -0.0005179, -0.0007907, 0.0008987
5: -0.0146023, -0.0072237, -0.0146264, -0.0078366, -0.0051379, 0.0058401
6: 0.0033743, 0.0052471, 0.0035299, 0.0052532, -0.0014823, 0.0013041
7: 0.0055927, 0.0104381, 0.0059952, 0.0104539, -0.0038351, 0.0033740
8: 0.0033770, 0.0059252, 0.0035887, 0.0059335, -0.0020168, 0.0017744
9: -0.0087344, -0.0057796, -0.0087440, -0.0060251, -0.0020575, 0.0023386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012320, upper bound: 0.0012649
time: 1.13 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0011002
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1_B2_B2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012320, upper bound: 0.0014427
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0011002
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A2_B2_B1

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

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
time: 1.23 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A2_B2_B2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014101, upper bound: 0.0013841
time: 1.20 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1_A1_A2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013737, upper bound: 0.0016601
time: 1.28 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013666, upper bound: 0.0016494
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1_A2_A2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007899, upper bound: 0.0014164
time: 1.24 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 23
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 21.35 seconds

### Candidate
type: A, layer: 3, pos: 23

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010512, upper bound: 0.0010021
time: 1.48 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006994, upper bound: 0.0007211
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2_A2_A1

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012544, upper bound: 0.0012629
time: 1.18 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2_A2_A2

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
time: 1.30 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0010859, 1.0054886, -0.0029585, 0.0031334
1: -0.0009695, 0.0001375, -0.0009934, 0.0001036, -0.0007372, 0.0007808
2: -0.0107827, -0.0049161, -0.0106033, -0.0047896, -0.0041377, 0.0039067
3: 0.0009645, 0.0036347, 0.0009069, 0.0035530, -0.0017782, 0.0018833
4: -0.0015591, -0.0004236, -0.0015244, -0.0003991, -0.0008008, 0.0007561
5: -0.0146023, -0.0072237, -0.0143767, -0.0070646, -0.0052041, 0.0049136
6: 0.0033743, 0.0052471, 0.0033339, 0.0051898, -0.0012471, 0.0013209
7: 0.0055927, 0.0104381, 0.0054882, 0.0102899, -0.0032267, 0.0034175
8: 0.0033770, 0.0059252, 0.0033220, 0.0058472, -0.0016969, 0.0017972
9: -0.0087344, -0.0057796, -0.0086440, -0.0057159, -0.0020840, 0.0019676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012248, upper bound: 0.0012543
time: 1.07 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 129
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 144
type: A, layer: 3, pos: 144
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 242
type: B, layer: 3, pos: 242

Time for candidate selection: 21.58 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009055, upper bound: 0.0011250
time: 1.20 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008065, upper bound: 0.0008722
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0012687, 1.0057063, -0.0032420, 0.0030383
1: -0.0009695, 0.0001375, -0.0009478, 0.0001579, -0.0008078, 0.0007571
2: -0.0107827, -0.0049161, -0.0108909, -0.0050311, -0.0040121, 0.0042810
3: 0.0009645, 0.0036347, 0.0010168, 0.0036839, -0.0019485, 0.0018261
4: -0.0015591, -0.0004236, -0.0015800, -0.0004459, -0.0007765, 0.0008286
5: -0.0146023, -0.0072237, -0.0147384, -0.0073683, -0.0050461, 0.0053843
6: 0.0033743, 0.0052471, 0.0034110, 0.0052816, -0.0013666, 0.0012808
7: 0.0055927, 0.0104381, 0.0056876, 0.0105275, -0.0035358, 0.0033137
8: 0.0033770, 0.0059252, 0.0034269, 0.0059722, -0.0018594, 0.0017427
9: -0.0087344, -0.0057796, -0.0087889, -0.0058375, -0.0020207, 0.0021561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012248, upper bound: 0.0012543
time: 1.11 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 145
type: B, layer: 3, pos: 145
type: A, layer: 3, pos: 129
type: B, layer: 3, pos: 129
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 144
type: B, layer: 3, pos: 144
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 242
type: A, layer: 3, pos: 242

Time for candidate selection: 21.75 seconds

### Candidate
type: B, layer: 3, pos: 23

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009055, upper bound: 0.0011250
time: 1.22 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008065, upper bound: 0.0008722
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1_B2_B1

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

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014989
time: 1.19 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013509, upper bound: 0.0014293
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1_B2_B2

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

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014313, upper bound: 0.0014427
time: 1.19 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013509, upper bound: 0.0014293
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A2_B2_B1

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

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
time: 1.37 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A2_B2_B2

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
time: 1.14 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
time: 1.17 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.61 seconds
IS_A2_B1_B1_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0014714, upper bound: 0.0016206
IS_A2_B1_B1_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013740, upper bound: 0.0016230
IS_A2_B1_B1_A1_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0008675, upper bound: 0.0011030
IS_A2_B1_B1_A1_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0008009, upper bound: 0.0009020
IS_A2_B1_B2_A1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013737, upper bound: 0.0016376
IS_A2_B1_B2_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013665, upper bound: 0.0016231
IS_A2_B2_B1_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013737, upper bound: 0.0016601
IS_A2_B2_B1_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013666, upper bound: 0.0016494
IS_A2_B2_B1_A1_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0009710, upper bound: 0.0010194
IS_A2_B2_B1_A1_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0008122, upper bound: 0.0009449
IS_A2_B2_B1_A1_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
IS_A2_B2_B1_A1_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
IS_A2_B2_B1_A1_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
IS_A2_B2_B1_A1_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
IS_A2_B2_B1_A2_A2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0015224, upper bound: 0.0015924
IS_A2_B2_B1_A2_A2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0014496, upper bound: 0.0015894
IS_A2_B2_B1_A2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0015224, upper bound: 0.0016341
IS_A2_B2_B1_A2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0014496, upper bound: 0.0016395
IS_A2_B2_B1_A2_A2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0012320, upper bound: 0.0012649
IS_A2_B2_B1_A2_A2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0011002
IS_A2_B2_B1_A2_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0012320, upper bound: 0.0014427
IS_A2_B2_B1_A2_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0011002
IS_A2_B2_B1_A2_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
IS_A2_B2_B1_A2_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
IS_A2_B2_B1_A2_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0014101, upper bound: 0.0013841
IS_A2_B2_B1_A2_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
IS_A2_B2_B2_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013737, upper bound: 0.0016601
IS_A2_B2_B2_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013666, upper bound: 0.0016494
IS_A2_B2_B2_A1_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0010512, upper bound: 0.0010021
IS_A2_B2_B2_A1_B2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0006994, upper bound: 0.0007211
IS_A2_B2_B2_A1_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0012544, upper bound: 0.0012629
IS_A2_B2_B2_A1_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
IS_A2_B2_B2_A1_B2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0011408, upper bound: 0.0013673
IS_A2_B2_B2_A1_B2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0011037, upper bound: 0.0012279
IS_A2_B2_B2_A2_A2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0009055, upper bound: 0.0011250
IS_A2_B2_B2_A2_A2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0008065, upper bound: 0.0008722
IS_A2_B2_B2_A2_A2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0009055, upper bound: 0.0011250
IS_A2_B2_B2_A2_A2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0008065, upper bound: 0.0008722
IS_A2_B2_B2_A2_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013536, upper bound: 0.0014989
IS_A2_B2_B2_A2_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013509, upper bound: 0.0014293
IS_A2_B2_B2_A2_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0014313, upper bound: 0.0014427
IS_A2_B2_B2_A2_A2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013509, upper bound: 0.0014293
IS_A2_B2_B2_A2_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
IS_A2_B2_B2_A2_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603
IS_A2_B2_B2_A2_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0014541
IS_A2_B2_B2_A2_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.61
Output dim: 0, lower bound: -0.0012974, upper bound: 0.0013603

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 1.0013121, 1.0057251, 1.0016490, 1.0056376, -0.0033111, 0.0030454
1: -0.0009370, 0.0001626, -0.0008531, 0.0001408, -0.0008250, 0.0007588
2: -0.0109158, -0.0050884, -0.0108002, -0.0055333, -0.0040215, 0.0043722
3: 0.0010429, 0.0036953, 0.0012454, 0.0036426, -0.0019900, 0.0018304
4: -0.0015848, -0.0004570, -0.0015625, -0.0005431, -0.0007783, 0.0008462
5: -0.0147698, -0.0074404, -0.0146243, -0.0080000, -0.0050580, 0.0054991
6: 0.0034293, 0.0052896, 0.0035713, 0.0052526, -0.0013957, 0.0012838
7: 0.0057350, 0.0105481, 0.0061024, 0.0104526, -0.0036112, 0.0033215
8: 0.0034518, 0.0059830, 0.0036451, 0.0059328, -0.0018991, 0.0017467
9: -0.0088014, -0.0058664, -0.0087432, -0.0060905, -0.0020254, 0.0022021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
time: 1.21 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 1.0013386, 1.0057207, 1.0016844, 1.0059094, -0.0036025, 0.0030570
1: -0.0009304, 0.0001615, -0.0008442, 0.0002085, -0.0008976, 0.0007617
2: -0.0109099, -0.0051234, -0.0111590, -0.0055800, -0.0040367, 0.0047571
3: 0.0010588, 0.0036926, 0.0012667, 0.0038060, -0.0021652, 0.0018373
4: -0.0015837, -0.0004637, -0.0016319, -0.0005521, -0.0007813, 0.0009207
5: -0.0147623, -0.0074844, -0.0150756, -0.0080588, -0.0050771, 0.0059831
6: 0.0034405, 0.0052877, 0.0035862, 0.0053672, -0.0015186, 0.0012886
7: 0.0057639, 0.0105432, 0.0061410, 0.0107489, -0.0039290, 0.0033341
8: 0.0034670, 0.0059804, 0.0036654, 0.0060886, -0.0020662, 0.0017534
9: -0.0087984, -0.0058840, -0.0089239, -0.0061140, -0.0020331, 0.0023959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016230
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016230
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2_B1_A2_A1

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

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
time: 1.23 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016230
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2_B1_A2_A2

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016206
time: 1.32 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016230
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1_B1_A2_A1

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016462
time: 1.23 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016494
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_B1_A1_B2_A1_B1_A2_A2

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

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016462
time: 1.23 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2_A1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016494
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 1.0011817, 1.0056244, 1.0015237, 1.0055317, -0.0033065, 0.0030434
1: -0.0009695, 0.0001375, -0.0008843, 0.0001144, -0.0008239, 0.0007583
2: -0.0107827, -0.0049161, -0.0106602, -0.0053677, -0.0040187, 0.0043661
3: 0.0009645, 0.0036347, 0.0011700, 0.0035789, -0.0019873, 0.0018292
4: -0.0015591, -0.0004236, -0.0015354, -0.0005110, -0.0007778, 0.0008451
5: -0.0146023, -0.0072237, -0.0144482, -0.0077917, -0.0050545, 0.0054915
6: 0.0033743, 0.0052471, 0.0035184, 0.0052079, -0.0013938, 0.0012829
7: 0.0055927, 0.0104381, 0.0059657, 0.0103369, -0.0036062, 0.0033192
8: 0.0033770, 0.0059252, 0.0035731, 0.0058719, -0.0018964, 0.0017456
9: -0.0087344, -0.0057796, -0.0086727, -0.0060071, -0.0020241, 0.0021990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016341
time: 1.19 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016341
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 1.0012120, 1.0056206, 1.0015628, 1.0057992, -0.0035717, 0.0030323
1: -0.0009620, 0.0001365, -0.0008745, 0.0001811, -0.0008900, 0.0007556
2: -0.0107776, -0.0049562, -0.0110135, -0.0054194, -0.0040041, 0.0047163
3: 0.0009827, 0.0036324, 0.0011936, 0.0037398, -0.0021467, 0.0018225
4: -0.0015581, -0.0004314, -0.0016038, -0.0005210, -0.0007750, 0.0009128
5: -0.0145959, -0.0072741, -0.0148927, -0.0078567, -0.0050361, 0.0059319
6: 0.0033871, 0.0052454, 0.0035350, 0.0053208, -0.0015056, 0.0012782
7: 0.0056258, 0.0104339, 0.0060084, 0.0106288, -0.0038954, 0.0033072
8: 0.0033944, 0.0059229, 0.0035956, 0.0060254, -0.0020486, 0.0017392
9: -0.0087318, -0.0057998, -0.0088506, -0.0060331, -0.0020167, 0.0023754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016395
time: 1.25 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016395
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 1.0013394, 1.0057197, 1.0011817, 1.0056244, -0.0029896, 0.0032529
1: -0.0009302, 0.0001612, -0.0009695, 0.0001375, -0.0007449, 0.0008105
2: -0.0109085, -0.0051244, -0.0107827, -0.0049161, -0.0042954, 0.0039477
3: 0.0010593, 0.0036919, 0.0009645, 0.0036347, -0.0017968, 0.0019551
4: -0.0015834, -0.0004639, -0.0015591, -0.0004236, -0.0008314, 0.0007641
5: -0.0147605, -0.0074857, -0.0146023, -0.0072237, -0.0054025, 0.0049652
6: 0.0034408, 0.0052872, 0.0033743, 0.0052471, -0.0012602, 0.0013712
7: 0.0057647, 0.0105420, 0.0055927, 0.0104381, -0.0032605, 0.0035478
8: 0.0034675, 0.0059798, 0.0033770, 0.0059252, -0.0017147, 0.0018657
9: -0.0087977, -0.0058846, -0.0087344, -0.0057796, -0.0021634, 0.0019883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016462
time: 1.25 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016494
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 1.0013813, 1.0059683, 1.0012120, 1.0056206, -0.0029991, 0.0035282
1: -0.0009198, 0.0002232, -0.0009620, 0.0001365, -0.0007473, 0.0008791
2: -0.0112368, -0.0051796, -0.0107776, -0.0049562, -0.0046589, 0.0039603
3: 0.0010844, 0.0038414, 0.0009827, 0.0036324, -0.0018026, 0.0021205
4: -0.0016470, -0.0004746, -0.0015581, -0.0004314, -0.0009017, 0.0007665
5: -0.0151735, -0.0075551, -0.0145959, -0.0072741, -0.0058597, 0.0049810
6: 0.0034584, 0.0053920, 0.0033871, 0.0052454, -0.0012642, 0.0014873
7: 0.0058103, 0.0108132, 0.0056258, 0.0104339, -0.0032710, 0.0038480
8: 0.0034914, 0.0061224, 0.0033944, 0.0059229, -0.0017202, 0.0020236
9: -0.0089631, -0.0059123, -0.0087318, -0.0057998, -0.0023465, 0.0019946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016462
time: 1.28 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016494
time: 1.28 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 3.87 seconds
IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
IS_A2_B1_B1_A1_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016230
IS_A2_B1_B1_A1_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016230
IS_A2_B1_B2_A1_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016206
IS_A2_B1_B2_A1_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016230
IS_A2_B1_B2_A1_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016206
IS_A2_B1_B2_A1_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016230
IS_A2_B2_B1_A1_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016462
IS_A2_B2_B1_A1_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016494
IS_A2_B2_B1_A1_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016462
IS_A2_B2_B1_A1_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016494
IS_A2_B2_B1_A2_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016341
IS_A2_B2_B1_A2_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016341
IS_A2_B2_B1_A2_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016395
IS_A2_B2_B1_A2_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015152, upper bound: 0.0016395
IS_A2_B2_B2_A1_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016462
IS_A2_B2_B2_A1_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0014879, upper bound: 0.0016494
IS_A2_B2_B2_A1_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016462
IS_A2_B2_B2_A1_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.87
Output dim: 0, lower bound: -0.0015069, upper bound: 0.0016494

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A1

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.16 + 598.68 = 601.84 seconds
