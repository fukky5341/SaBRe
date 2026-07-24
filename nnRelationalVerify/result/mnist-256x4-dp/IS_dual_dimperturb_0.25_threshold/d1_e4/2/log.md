## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 6.43e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0000332, 0.0008493, -0.0000332, 0.0008493, -0.0005122, 0.0005122)
1: (-0.0034917, -0.0033259, -0.0034917, -0.0033259, -0.0000619, 0.0000619)
2: (0.0148935, 0.0160044, 0.0148935, 0.0160044, -0.0006101, 0.0006101)
3: (1.0066880, 1.0069833, 1.0066880, 1.0069833, -0.0002074, 0.0002074)
4: (-0.0042510, -0.0040808, -0.0042510, -0.0040808, -0.0000870, 0.0000870)
5: (0.0039544, 0.0046281, 0.0039544, 0.0046281, -0.0003882, 0.0003882)
6: (-0.0026076, -0.0025632, -0.0026076, -0.0025632, -0.0000387, 0.0000387)
7: (-0.0129980, -0.0113500, -0.0129980, -0.0113500, -0.0011456, 0.0011456)
8: (-0.0137135, -0.0119442, -0.0137135, -0.0119442, -0.0008597, 0.0008597)
9: (0.0018025, 0.0026383, 0.0018025, 0.0026383, -0.0003794, 0.0003794)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 1.37 = 2.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0000861, upper bound: 0.0000862

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000816, upper bound: 0.0000725
time: 0.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000816, upper bound: 0.0000816
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.40 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 3, lower bound: -0.0000816, upper bound: 0.0000725
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 3, lower bound: -0.0000816, upper bound: 0.0000816

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0000416, 0.0008278, -0.0000332, 0.0008398, -0.0004916, 0.0004813
1: -0.0034866, -0.0033266, -0.0034895, -0.0033259, -0.0000583, 0.0000600
2: 0.0148855, 0.0159840, 0.0148935, 0.0159957, -0.0005910, 0.0005809
3: 1.0066999, 1.0069489, 1.0066880, 1.0069686, -0.0001775, 0.0001714
4: -0.0042490, -0.0040800, -0.0042501, -0.0040808, -0.0000842, 0.0000852
5: 0.0039482, 0.0046124, 0.0039544, 0.0046212, -0.0003731, 0.0003655
6: -0.0026069, -0.0025682, -0.0026076, -0.0025655, -0.0000343, 0.0000329
7: -0.0129177, -0.0113301, -0.0129635, -0.0113500, -0.0010305, 0.0010648
8: -0.0137016, -0.0119382, -0.0137085, -0.0119442, -0.0008427, 0.0008492
9: 0.0018020, 0.0026374, 0.0018025, 0.0026380, -0.0003786, 0.0003781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 189

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000646
time: 0.58 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000606
time: 0.55 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0000332, 0.0008351, -0.0000332, 0.0008444, -0.0005097, 0.0004726
1: -0.0034898, -0.0033259, -0.0034910, -0.0033259, -0.0000590, 0.0000612
2: 0.0148935, 0.0159899, 0.0148935, 0.0159994, -0.0006074, 0.0005724
3: 1.0066880, 1.0069764, 1.0066880, 1.0069807, -0.0002059, 0.0001646
4: -0.0042495, -0.0040808, -0.0042505, -0.0040808, -0.0000834, 0.0000867
5: 0.0039544, 0.0046177, 0.0039544, 0.0046245, -0.0003863, 0.0003590
6: -0.0026076, -0.0025645, -0.0026076, -0.0025637, -0.0000384, 0.0000319
7: -0.0129545, -0.0113500, -0.0129829, -0.0113500, -0.0009983, 0.0011383
8: -0.0137040, -0.0119442, -0.0137103, -0.0119442, -0.0008380, 0.0008577
9: 0.0018025, 0.0026376, 0.0018025, 0.0026381, -0.0003792, 0.0003777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 189
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 189

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000720
time: 0.56 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000721
time: 0.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.26 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000646
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000606
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 3, lower bound: -0.0000714, upper bound: 0.0000720
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 3, lower bound: -0.0000719, upper bound: 0.0000721

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0000493, 0.0006848, -0.0000332, 0.0007840, -0.0003989, 0.0003220
1: -0.0034872, -0.0033295, -0.0034895, -0.0033270, -0.0000574, 0.0000569
2: 0.0148733, 0.0158216, 0.0148935, 0.0159323, -0.0004860, 0.0003993
3: 1.0066675, 1.0069184, 1.0066880, 1.0069548, -0.0001525, 0.0001182
4: -0.0042275, -0.0040779, -0.0042418, -0.0040808, -0.0000600, 0.0000715
5: 0.0039422, 0.0045046, 0.0039544, 0.0045792, -0.0003032, 0.0002453
6: -0.0026072, -0.0025812, -0.0026076, -0.0025712, -0.0000263, 0.0000184
7: -0.0125543, -0.0113227, -0.0128143, -0.0113500, -0.0006301, 0.0008310
8: -0.0135028, -0.0119167, -0.0136313, -0.0119442, -0.0006178, 0.0007235
9: 0.0017917, 0.0025592, 0.0018025, 0.0026074, -0.0003301, 0.0002888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000602
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000603
time: 0.61 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0000416, 0.0008138, -0.0000332, 0.0008383, -0.0004913, 0.0003320
1: -0.0034866, -0.0033270, -0.0034895, -0.0033260, -0.0000583, 0.0000574
2: 0.0148855, 0.0159657, 0.0148935, 0.0159940, -0.0005906, 0.0004080
3: 1.0066999, 1.0069375, 1.0066880, 1.0069675, -0.0001773, 0.0001209
4: -0.0042465, -0.0040800, -0.0042499, -0.0040808, -0.0000607, 0.0000852
5: 0.0039482, 0.0046017, 0.0039544, 0.0046201, -0.0003728, 0.0002527
6: -0.0026069, -0.0025688, -0.0026076, -0.0025655, -0.0000343, 0.0000210
7: -0.0128889, -0.0113301, -0.0129608, -0.0113500, -0.0006770, 0.0010643
8: -0.0136772, -0.0119382, -0.0137063, -0.0119442, -0.0006223, 0.0008485
9: 0.0018020, 0.0026271, 0.0018025, 0.0026370, -0.0003783, 0.0002907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000602
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000607
time: 0.53 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0000397, 0.0006911, -0.0000332, 0.0007881, -0.0004237, 0.0003132
1: -0.0034905, -0.0033288, -0.0034910, -0.0033270, -0.0000580, 0.0000584
2: 0.0148829, 0.0158270, 0.0148935, 0.0159358, -0.0005078, 0.0003908
3: 1.0066613, 1.0069410, 1.0066880, 1.0069661, -0.0001865, 0.0001114
4: -0.0042279, -0.0040790, -0.0042421, -0.0040808, -0.0000593, 0.0000731
5: 0.0039493, 0.0045092, 0.0039544, 0.0045822, -0.0003214, 0.0002389
6: -0.0026082, -0.0025774, -0.0026076, -0.0025694, -0.0000319, 0.0000174
7: -0.0125833, -0.0113502, -0.0128338, -0.0113500, -0.0005980, 0.0009340
8: -0.0135044, -0.0119240, -0.0136332, -0.0119442, -0.0006131, 0.0007317
9: 0.0017924, 0.0025592, 0.0018025, 0.0026074, -0.0003305, 0.0002884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000601, upper bound: 0.0000721
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000721
time: 0.61 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0000332, 0.0008208, -0.0000332, 0.0008431, -0.0005094, 0.0003175
1: -0.0034898, -0.0033264, -0.0034910, -0.0033260, -0.0000590, 0.0000592
2: 0.0148935, 0.0159725, 0.0148935, 0.0159977, -0.0006070, 0.0003947
3: 1.0066880, 1.0069662, 1.0066880, 1.0069798, -0.0002056, 0.0001120
4: -0.0042469, -0.0040808, -0.0042502, -0.0040808, -0.0000597, 0.0000866
5: 0.0039544, 0.0046068, 0.0039544, 0.0046236, -0.0003861, 0.0002420
6: -0.0026076, -0.0025654, -0.0026076, -0.0025638, -0.0000384, 0.0000187
7: -0.0129259, -0.0113500, -0.0129801, -0.0113500, -0.0006194, 0.0011372
8: -0.0136797, -0.0119442, -0.0137081, -0.0119442, -0.0006178, 0.0008571
9: 0.0018025, 0.0026273, 0.0018025, 0.0026372, -0.0003789, 0.0002905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000714
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000720
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.33 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000602
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000603
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000602
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000607
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 3, lower bound: -0.0000601, upper bound: 0.0000721
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000721
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000714
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 3, lower bound: -0.0000706, upper bound: 0.0000720

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0000493, 0.0006848, -0.0000397, 0.0006970, -0.0003088, 0.0003049
1: -0.0034872, -0.0033295, -0.0034901, -0.0033288, -0.0000556, 0.0000571
2: 0.0148733, 0.0158216, 0.0148829, 0.0158332, -0.0003829, 0.0003782
3: 1.0066675, 1.0069184, 1.0066613, 1.0069364, -0.0001233, 0.0001227
4: -0.0042275, -0.0040779, -0.0042286, -0.0040790, -0.0000570, 0.0000577
5: 0.0039422, 0.0045046, 0.0039493, 0.0045136, -0.0002352, 0.0002323
6: -0.0026072, -0.0025812, -0.0026082, -0.0025786, -0.0000184, 0.0000181
7: -0.0125543, -0.0113227, -0.0125992, -0.0113502, -0.0005993, 0.0006091
8: -0.0135028, -0.0119167, -0.0135091, -0.0119240, -0.0005886, 0.0005947
9: 0.0017917, 0.0025592, 0.0017924, 0.0025597, -0.0002788, 0.0002782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000646
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000646
time: 0.59 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0000493, 0.0006848, -0.0000332, 0.0008261, -0.0004661, 0.0003220
1: -0.0034872, -0.0033295, -0.0034895, -0.0033264, -0.0000580, 0.0000569
2: 0.0148733, 0.0158216, 0.0148935, 0.0159774, -0.0005618, 0.0003993
3: 1.0066675, 1.0069184, 1.0066880, 1.0069574, -0.0001758, 0.0001182
4: -0.0042275, -0.0040779, -0.0042476, -0.0040808, -0.0000600, 0.0000814
5: 0.0039422, 0.0045046, 0.0039544, 0.0046107, -0.0003538, 0.0002453
6: -0.0026072, -0.0025812, -0.0026076, -0.0025660, -0.0000325, 0.0000184
7: -0.0125543, -0.0113227, -0.0129353, -0.0113500, -0.0006301, 0.0010027
8: -0.0135028, -0.0119167, -0.0136841, -0.0119442, -0.0006178, 0.0008142
9: 0.0017917, 0.0025592, 0.0018025, 0.0026277, -0.0003653, 0.0002888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000645
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000645
time: 0.61 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0000416, 0.0008138, -0.0000397, 0.0006970, -0.0003317, 0.0004614
1: -0.0034866, -0.0033270, -0.0034901, -0.0033288, -0.0000555, 0.0000599
2: 0.0148855, 0.0159657, 0.0148829, 0.0158332, -0.0004087, 0.0005562
3: 1.0066999, 1.0069375, 1.0066613, 1.0069364, -0.0001234, 0.0001749
4: -0.0042465, -0.0040800, -0.0042286, -0.0040790, -0.0000805, 0.0000610
5: 0.0039482, 0.0046017, 0.0039493, 0.0045136, -0.0002525, 0.0003503
6: -0.0026069, -0.0025688, -0.0026082, -0.0025786, -0.0000200, 0.0000324
7: -0.0128889, -0.0113301, -0.0125992, -0.0113502, -0.0009954, 0.0006661
8: -0.0136772, -0.0119382, -0.0135091, -0.0119240, -0.0008071, 0.0006236
9: 0.0018020, 0.0026271, 0.0017924, 0.0025597, -0.0002892, 0.0003647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000602
time: 0.57 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000602
time: 0.56 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0000416, 0.0008138, -0.0000332, 0.0008261, -0.0003359, 0.0003320
1: -0.0034866, -0.0033270, -0.0034895, -0.0033264, -0.0000560, 0.0000574
2: 0.0148855, 0.0159657, 0.0148935, 0.0159774, -0.0004127, 0.0004080
3: 1.0066999, 1.0069375, 1.0066880, 1.0069574, -0.0001215, 0.0001209
4: -0.0042465, -0.0040800, -0.0042476, -0.0040808, -0.0000607, 0.0000614
5: 0.0039482, 0.0046017, 0.0039544, 0.0046107, -0.0002556, 0.0002527
6: -0.0026069, -0.0025688, -0.0026076, -0.0025660, -0.0000214, 0.0000210
7: -0.0128889, -0.0113301, -0.0129353, -0.0113500, -0.0006770, 0.0006868
8: -0.0136772, -0.0119382, -0.0136841, -0.0119442, -0.0006223, 0.0006283
9: 0.0018020, 0.0026271, 0.0018025, 0.0026277, -0.0002913, 0.0002907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000605
time: 0.57 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000604
time: 0.62 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0000397, 0.0006911, -0.0000416, 0.0007717, -0.0003940, 0.0003405
1: -0.0034905, -0.0033288, -0.0034866, -0.0033277, -0.0000591, 0.0000555
2: 0.0148829, 0.0158270, 0.0148855, 0.0159208, -0.0004802, 0.0004154
3: 1.0066613, 1.0069410, 1.0066999, 1.0069362, -0.0001503, 0.0001390
4: -0.0042279, -0.0040790, -0.0042406, -0.0040800, -0.0000614, 0.0000706
5: 0.0039493, 0.0045092, 0.0039482, 0.0045702, -0.0002996, 0.0002588
6: -0.0026082, -0.0025774, -0.0026069, -0.0025740, -0.0000259, 0.0000225
7: -0.0125833, -0.0113502, -0.0127683, -0.0113301, -0.0007076, 0.0008202
8: -0.0135044, -0.0119240, -0.0136243, -0.0119382, -0.0006246, 0.0007163
9: 0.0017924, 0.0025592, 0.0018020, 0.0026069, -0.0003294, 0.0002891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000601, upper bound: 0.0000721
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000601, upper bound: 0.0000721
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0000397, 0.0006911, -0.0000332, 0.0007784, -0.0003794, 0.0003132
1: -0.0034905, -0.0033288, -0.0034898, -0.0033270, -0.0000580, 0.0000559
2: 0.0148829, 0.0158270, 0.0148935, 0.0159262, -0.0004670, 0.0003908
3: 1.0066613, 1.0069410, 1.0066880, 1.0069611, -0.0001389, 0.0001114
4: -0.0042279, -0.0040790, -0.0042411, -0.0040808, -0.0000593, 0.0000697
5: 0.0039493, 0.0045092, 0.0039544, 0.0045750, -0.0002888, 0.0002389
6: -0.0026082, -0.0025774, -0.0026076, -0.0025702, -0.0000236, 0.0000174
7: -0.0125833, -0.0113502, -0.0128047, -0.0113500, -0.0005980, 0.0007618
8: -0.0135044, -0.0119240, -0.0136269, -0.0119442, -0.0006131, 0.0007119
9: 0.0017924, 0.0025592, 0.0018025, 0.0026069, -0.0003292, 0.0002884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 189

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000721
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000720
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0000332, 0.0008208, -0.0000397, 0.0007009, -0.0003508, 0.0004468
1: -0.0034898, -0.0033264, -0.0034917, -0.0033288, -0.0000559, 0.0000612
2: 0.0148935, 0.0159725, 0.0148829, 0.0158368, -0.0004257, 0.0005430
3: 1.0066880, 1.0069662, 1.0066613, 1.0069462, -0.0001539, 0.0001636
4: -0.0042469, -0.0040808, -0.0042289, -0.0040790, -0.0000796, 0.0000624
5: 0.0039544, 0.0046068, 0.0039493, 0.0045166, -0.0002665, 0.0003395
6: -0.0026076, -0.0025654, -0.0026082, -0.0025767, -0.0000246, 0.0000301
7: -0.0129259, -0.0113500, -0.0126153, -0.0113502, -0.0009370, 0.0007451
8: -0.0136797, -0.0119442, -0.0135108, -0.0119240, -0.0008027, 0.0006315
9: 0.0018025, 0.0026273, 0.0017924, 0.0025598, -0.0002897, 0.0003644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000603, upper bound: 0.0000715
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000603, upper bound: 0.0000715
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0000332, 0.0008208, -0.0000332, 0.0008303, -0.0003609, 0.0003175
1: -0.0034898, -0.0033264, -0.0034910, -0.0033264, -0.0000564, 0.0000592
2: 0.0148935, 0.0159725, 0.0148935, 0.0159816, -0.0004344, 0.0003947
3: 1.0066880, 1.0069662, 1.0066880, 1.0069705, -0.0001566, 0.0001120
4: -0.0042469, -0.0040808, -0.0042479, -0.0040808, -0.0000597, 0.0000631
5: 0.0039544, 0.0046068, 0.0039544, 0.0046138, -0.0002738, 0.0002420
6: -0.0026076, -0.0025654, -0.0026076, -0.0025644, -0.0000272, 0.0000187
7: -0.0129259, -0.0113500, -0.0129556, -0.0113500, -0.0006194, 0.0007920
8: -0.0136797, -0.0119442, -0.0136859, -0.0119442, -0.0006178, 0.0006359
9: 0.0018025, 0.0026273, 0.0018025, 0.0026278, -0.0002917, 0.0002905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000603, upper bound: 0.0000720
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000603, upper bound: 0.0000721
time: 0.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.32 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000646
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000646
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000645
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000645
IS_A1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000602
IS_A1_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000602
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000605
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000634, upper bound: 0.0000604
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000601, upper bound: 0.0000721
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000601, upper bound: 0.0000721
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000721
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000602, upper bound: 0.0000720
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000603, upper bound: 0.0000715
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000603, upper bound: 0.0000715
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000603, upper bound: 0.0000720
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 3, lower bound: -0.0000603, upper bound: 0.0000721

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0000493, 0.0006848, -0.0000493, 0.0006848, -0.0002921, 0.0002921
1: -0.0034872, -0.0033295, -0.0034872, -0.0033295, -0.0000551, 0.0000551
2: 0.0148733, 0.0158216, 0.0148733, 0.0158216, -0.0003672, 0.0003672
3: 1.0066675, 1.0069184, 1.0066675, 1.0069184, -0.0001044, 0.0001044
4: -0.0042275, -0.0040779, -0.0042275, -0.0040779, -0.0000562, 0.0000562
5: 0.0039422, 0.0045046, 0.0039422, 0.0045046, -0.0002229, 0.0002229
6: -0.0026072, -0.0025812, -0.0026072, -0.0025812, -0.0000150, 0.0000150
7: -0.0125543, -0.0113227, -0.0125543, -0.0113227, -0.0005434, 0.0005434
8: -0.0135028, -0.0119167, -0.0135028, -0.0119167, -0.0005858, 0.0005858
9: 0.0017917, 0.0025592, 0.0017917, 0.0025592, -0.0002781, 0.0002781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000591, upper bound: 0.0000576
time: 0.56 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000591, upper bound: 0.0000579
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0000493, 0.0006848, -0.0000397, 0.0006911, -0.0003175, 0.0003049
1: -0.0034872, -0.0033295, -0.0034905, -0.0033288, -0.0000556, 0.0000572
2: 0.0148733, 0.0158216, 0.0148829, 0.0158270, -0.0003896, 0.0003782
3: 1.0066675, 1.0069184, 1.0066613, 1.0069410, -0.0001389, 0.0001227
4: -0.0042275, -0.0040779, -0.0042279, -0.0040790, -0.0000570, 0.0000581
5: 0.0039422, 0.0045046, 0.0039493, 0.0045092, -0.0002415, 0.0002323
6: -0.0026072, -0.0025812, -0.0026082, -0.0025774, -0.0000209, 0.0000181
7: -0.0125543, -0.0113227, -0.0125833, -0.0113502, -0.0005993, 0.0006506
8: -0.0135028, -0.0119167, -0.0135044, -0.0119240, -0.0005886, 0.0005957
9: 0.0017917, 0.0025592, 0.0017924, 0.0025592, -0.0002787, 0.0002782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000591, upper bound: 0.0000576
time: 0.55 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000591, upper bound: 0.0000578
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0000493, 0.0006848, -0.0000416, 0.0008138, -0.0004486, 0.0003151
1: -0.0034872, -0.0033295, -0.0034866, -0.0033270, -0.0000579, 0.0000549
2: 0.0148733, 0.0158216, 0.0148855, 0.0159657, -0.0005452, 0.0003930
3: 1.0066675, 1.0069184, 1.0066999, 1.0069375, -0.0001567, 0.0001046
4: -0.0042275, -0.0040779, -0.0042465, -0.0040800, -0.0000595, 0.0000798
5: 0.0039422, 0.0045046, 0.0039482, 0.0046017, -0.0003409, 0.0002403
6: -0.0026072, -0.0025812, -0.0026069, -0.0025688, -0.0000293, 0.0000166
7: -0.0125543, -0.0113227, -0.0128889, -0.0113301, -0.0006004, 0.0009395
8: -0.0135028, -0.0119167, -0.0136772, -0.0119382, -0.0006147, 0.0008043
9: 0.0017917, 0.0025592, 0.0018020, 0.0026271, -0.0003646, 0.0002886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000558, upper bound: 0.0000576
time: 0.58 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000578
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0000493, 0.0006848, -0.0000332, 0.0008208, -0.0004713, 0.0003220
1: -0.0034872, -0.0033295, -0.0034898, -0.0033264, -0.0000580, 0.0000567
2: 0.0148733, 0.0158216, 0.0148935, 0.0159725, -0.0005662, 0.0003993
3: 1.0066675, 1.0069184, 1.0066880, 1.0069662, -0.0001866, 0.0001182
4: -0.0042275, -0.0040779, -0.0042469, -0.0040808, -0.0000600, 0.0000817
5: 0.0039422, 0.0045046, 0.0039544, 0.0046068, -0.0003576, 0.0002453
6: -0.0026072, -0.0025812, -0.0026076, -0.0025654, -0.0000339, 0.0000184
7: -0.0125543, -0.0113227, -0.0129259, -0.0113500, -0.0006301, 0.0010257
8: -0.0135028, -0.0119167, -0.0136797, -0.0119442, -0.0006178, 0.0008156
9: 0.0017917, 0.0025592, 0.0018025, 0.0026273, -0.0003654, 0.0002888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000558, upper bound: 0.0000576
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000579
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0000397, 0.0006911, -0.0000493, 0.0006848, -0.0003049, 0.0003175
1: -0.0034905, -0.0033288, -0.0034872, -0.0033295, -0.0000572, 0.0000556
2: 0.0148829, 0.0158270, 0.0148733, 0.0158216, -0.0003782, 0.0003896
3: 1.0066613, 1.0069410, 1.0066675, 1.0069184, -0.0001227, 0.0001389
4: -0.0042279, -0.0040790, -0.0042275, -0.0040779, -0.0000581, 0.0000570
5: 0.0039493, 0.0045092, 0.0039422, 0.0045046, -0.0002323, 0.0002415
6: -0.0026082, -0.0025774, -0.0026072, -0.0025812, -0.0000181, 0.0000209
7: -0.0125833, -0.0113502, -0.0125543, -0.0113227, -0.0006506, 0.0005993
8: -0.0135044, -0.0119240, -0.0135028, -0.0119167, -0.0005957, 0.0005886
9: 0.0017924, 0.0025592, 0.0017917, 0.0025592, -0.0002782, 0.0002787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000674
time: 0.64 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000619
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0000397, 0.0006911, -0.0000416, 0.0008138, -0.0004614, 0.0003405
1: -0.0034905, -0.0033288, -0.0034866, -0.0033270, -0.0000600, 0.0000555
2: 0.0148829, 0.0158270, 0.0148855, 0.0159657, -0.0005562, 0.0004154
3: 1.0066613, 1.0069410, 1.0066999, 1.0069375, -0.0001749, 0.0001390
4: -0.0042279, -0.0040790, -0.0042465, -0.0040800, -0.0000614, 0.0000805
5: 0.0039493, 0.0045092, 0.0039482, 0.0046017, -0.0003503, 0.0002588
6: -0.0026082, -0.0025774, -0.0026069, -0.0025688, -0.0000324, 0.0000225
7: -0.0125833, -0.0113502, -0.0128889, -0.0113301, -0.0007076, 0.0009954
8: -0.0135044, -0.0119240, -0.0136772, -0.0119382, -0.0006246, 0.0008071
9: 0.0017924, 0.0025592, 0.0018020, 0.0026271, -0.0003647, 0.0002891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000674
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000619
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0000397, 0.0006911, -0.0000397, 0.0006911, -0.0002902, 0.0002902
1: -0.0034905, -0.0033288, -0.0034905, -0.0033288, -0.0000561, 0.0000561
2: 0.0148829, 0.0158270, 0.0148829, 0.0158270, -0.0003650, 0.0003650
3: 1.0066613, 1.0069410, 1.0066613, 1.0069410, -0.0001113, 0.0001113
4: -0.0042279, -0.0040790, -0.0042279, -0.0040790, -0.0000560, 0.0000560
5: 0.0039493, 0.0045092, 0.0039493, 0.0045092, -0.0002215, 0.0002215
6: -0.0026082, -0.0025774, -0.0026082, -0.0025774, -0.0000158, 0.0000158
7: -0.0125833, -0.0113502, -0.0125833, -0.0113502, -0.0005410, 0.0005410
8: -0.0135044, -0.0119240, -0.0135044, -0.0119240, -0.0005842, 0.0005842
9: 0.0017924, 0.0025592, 0.0017924, 0.0025592, -0.0002780, 0.0002780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000674
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000619
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0000397, 0.0006911, -0.0000332, 0.0008208, -0.0004468, 0.0003132
1: -0.0034905, -0.0033288, -0.0034898, -0.0033264, -0.0000589, 0.0000559
2: 0.0148829, 0.0158270, 0.0148935, 0.0159725, -0.0005430, 0.0003908
3: 1.0066613, 1.0069410, 1.0066880, 1.0069662, -0.0001636, 0.0001114
4: -0.0042279, -0.0040790, -0.0042469, -0.0040808, -0.0000593, 0.0000796
5: 0.0039493, 0.0045092, 0.0039544, 0.0046068, -0.0003395, 0.0002389
6: -0.0026082, -0.0025774, -0.0026076, -0.0025654, -0.0000301, 0.0000174
7: -0.0125833, -0.0113502, -0.0129259, -0.0113500, -0.0005980, 0.0009370
8: -0.0135044, -0.0119240, -0.0136797, -0.0119442, -0.0006131, 0.0008027
9: 0.0017924, 0.0025592, 0.0018025, 0.0026273, -0.0003644, 0.0002884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000674
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000619
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0000332, 0.0008208, -0.0000493, 0.0006848, -0.0003220, 0.0004713
1: -0.0034898, -0.0033264, -0.0034872, -0.0033295, -0.0000567, 0.0000580
2: 0.0148935, 0.0159725, 0.0148733, 0.0158216, -0.0003993, 0.0005662
3: 1.0066880, 1.0069662, 1.0066675, 1.0069184, -0.0001182, 0.0001866
4: -0.0042469, -0.0040808, -0.0042275, -0.0040779, -0.0000817, 0.0000600
5: 0.0039544, 0.0046068, 0.0039422, 0.0045046, -0.0002453, 0.0003576
6: -0.0026076, -0.0025654, -0.0026072, -0.0025812, -0.0000184, 0.0000339
7: -0.0129259, -0.0113500, -0.0125543, -0.0113227, -0.0010258, 0.0006301
8: -0.0136797, -0.0119442, -0.0135028, -0.0119167, -0.0008156, 0.0006178
9: 0.0018025, 0.0026273, 0.0017917, 0.0025592, -0.0002888, 0.0003654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000678
time: 0.57 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000593
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0000332, 0.0008208, -0.0000397, 0.0006911, -0.0003132, 0.0004468
1: -0.0034898, -0.0033264, -0.0034905, -0.0033288, -0.0000559, 0.0000589
2: 0.0148935, 0.0159725, 0.0148829, 0.0158270, -0.0003908, 0.0005430
3: 1.0066880, 1.0069662, 1.0066613, 1.0069410, -0.0001114, 0.0001636
4: -0.0042469, -0.0040808, -0.0042279, -0.0040790, -0.0000796, 0.0000593
5: 0.0039544, 0.0046068, 0.0039493, 0.0045092, -0.0002389, 0.0003395
6: -0.0026076, -0.0025654, -0.0026082, -0.0025774, -0.0000174, 0.0000301
7: -0.0129259, -0.0113500, -0.0125833, -0.0113502, -0.0009370, 0.0005980
8: -0.0136797, -0.0119442, -0.0135044, -0.0119240, -0.0008027, 0.0006131
9: 0.0018025, 0.0026273, 0.0017924, 0.0025592, -0.0002884, 0.0003644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000677
time: 0.58 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000593
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0000332, 0.0008208, -0.0000416, 0.0008138, -0.0003320, 0.0003446
1: -0.0034898, -0.0033264, -0.0034866, -0.0033270, -0.0000576, 0.0000560
2: 0.0148935, 0.0159725, 0.0148855, 0.0159657, -0.0004080, 0.0004194
3: 1.0066880, 1.0069662, 1.0066999, 1.0069375, -0.0001209, 0.0001371
4: -0.0042469, -0.0040808, -0.0042465, -0.0040800, -0.0000619, 0.0000607
5: 0.0039544, 0.0046068, 0.0039482, 0.0046017, -0.0002527, 0.0002619
6: -0.0026076, -0.0025654, -0.0026069, -0.0025688, -0.0000210, 0.0000239
7: -0.0129259, -0.0113500, -0.0128889, -0.0113301, -0.0007283, 0.0006770
8: -0.0136797, -0.0119442, -0.0136772, -0.0119382, -0.0006293, 0.0006223
9: 0.0018025, 0.0026273, 0.0018020, 0.0026271, -0.0002907, 0.0002912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000685
time: 0.57 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000619
time: 0.57 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0000332, 0.0008208, -0.0000332, 0.0008208, -0.0003175, 0.0003175
1: -0.0034898, -0.0033264, -0.0034898, -0.0033264, -0.0000564, 0.0000564
2: 0.0148935, 0.0159725, 0.0148935, 0.0159725, -0.0003947, 0.0003947
3: 1.0066880, 1.0069662, 1.0066880, 1.0069662, -0.0001120, 0.0001120
4: -0.0042469, -0.0040808, -0.0042469, -0.0040808, -0.0000597, 0.0000597
5: 0.0039544, 0.0046068, 0.0039544, 0.0046068, -0.0002420, 0.0002420
6: -0.0026076, -0.0025654, -0.0026076, -0.0025654, -0.0000187, 0.0000187
7: -0.0129259, -0.0113500, -0.0129259, -0.0113500, -0.0006194, 0.0006194
8: -0.0136797, -0.0119442, -0.0136797, -0.0119442, -0.0006178, 0.0006178
9: 0.0018025, 0.0026273, 0.0018025, 0.0026273, -0.0002905, 0.0002905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000685
time: 0.57 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000619
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.29 seconds
IS_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000591, upper bound: 0.0000576
IS_A1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000591, upper bound: 0.0000579
IS_A1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000591, upper bound: 0.0000576
IS_A1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000591, upper bound: 0.0000578
IS_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000558, upper bound: 0.0000576
IS_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000578
IS_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000558, upper bound: 0.0000576
IS_A1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000562, upper bound: 0.0000579
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000674
IS_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000619
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000674
IS_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000619
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000674
IS_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000619
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000674
IS_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000538, upper bound: 0.0000619
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000678
IS_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000593
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000677
IS_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000578, upper bound: 0.0000593
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000685
IS_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000619
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000685
IS_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 3, lower bound: -0.0000550, upper bound: 0.0000619

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0000319, 0.0006911, -0.0000493, 0.0006848, -0.0002907, 0.0003175
1: -0.0034883, -0.0033288, -0.0034872, -0.0033295, -0.0000551, 0.0000556
2: 0.0148938, 0.0158270, 0.0148733, 0.0158216, -0.0003580, 0.0003896
3: 1.0066613, 1.0069375, 1.0066675, 1.0069184, -0.0001227, 0.0001368
4: -0.0042279, -0.0040808, -0.0042275, -0.0040779, -0.0000581, 0.0000536
5: 0.0039554, 0.0045092, 0.0039422, 0.0045046, -0.0002213, 0.0002415
6: -0.0026082, -0.0025776, -0.0026072, -0.0025812, -0.0000181, 0.0000208
7: -0.0125833, -0.0113565, -0.0125543, -0.0113227, -0.0006506, 0.0005881
8: -0.0135044, -0.0119442, -0.0135028, -0.0119167, -0.0005957, 0.0005528
9: 0.0018022, 0.0025592, 0.0017917, 0.0025592, -0.0002618, 0.0002787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000619
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000619
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0000319, 0.0006911, -0.0000416, 0.0008138, -0.0004472, 0.0003405
1: -0.0034883, -0.0033288, -0.0034866, -0.0033270, -0.0000579, 0.0000555
2: 0.0148938, 0.0158270, 0.0148855, 0.0159657, -0.0005360, 0.0004154
3: 1.0066613, 1.0069375, 1.0066999, 1.0069375, -0.0001749, 0.0001370
4: -0.0042279, -0.0040808, -0.0042465, -0.0040800, -0.0000614, 0.0000772
5: 0.0039554, 0.0045092, 0.0039482, 0.0046017, -0.0003392, 0.0002588
6: -0.0026082, -0.0025776, -0.0026069, -0.0025688, -0.0000324, 0.0000224
7: -0.0125833, -0.0113565, -0.0128889, -0.0113301, -0.0007076, 0.0009842
8: -0.0135044, -0.0119442, -0.0136772, -0.0119382, -0.0006246, 0.0007714
9: 0.0018022, 0.0025592, 0.0018020, 0.0026271, -0.0003483, 0.0002891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000619
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000619
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0000319, 0.0006911, -0.0000397, 0.0006911, -0.0002787, 0.0002902
1: -0.0034883, -0.0033288, -0.0034905, -0.0033288, -0.0000523, 0.0000561
2: 0.0148938, 0.0158270, 0.0148829, 0.0158270, -0.0003485, 0.0003650
3: 1.0066613, 1.0069375, 1.0066613, 1.0069410, -0.0001113, 0.0001067
4: -0.0042279, -0.0040808, -0.0042279, -0.0040790, -0.0000560, 0.0000532
5: 0.0039554, 0.0045092, 0.0039493, 0.0045092, -0.0002126, 0.0002215
6: -0.0026082, -0.0025776, -0.0026082, -0.0025774, -0.0000158, 0.0000157
7: -0.0125833, -0.0113565, -0.0125833, -0.0113502, -0.0005410, 0.0005315
8: -0.0135044, -0.0119442, -0.0135044, -0.0119240, -0.0005842, 0.0005525
9: 0.0018022, 0.0025592, 0.0017924, 0.0025592, -0.0002619, 0.0002780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000619
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000619
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0000319, 0.0006911, -0.0000332, 0.0008208, -0.0004352, 0.0003132
1: -0.0034883, -0.0033288, -0.0034898, -0.0033264, -0.0000551, 0.0000559
2: 0.0148938, 0.0158270, 0.0148935, 0.0159725, -0.0005265, 0.0003908
3: 1.0066613, 1.0069375, 1.0066880, 1.0069662, -0.0001636, 0.0001068
4: -0.0042279, -0.0040808, -0.0042469, -0.0040808, -0.0000593, 0.0000768
5: 0.0039554, 0.0045092, 0.0039544, 0.0046068, -0.0003305, 0.0002389
6: -0.0026082, -0.0025776, -0.0026076, -0.0025654, -0.0000301, 0.0000173
7: -0.0125833, -0.0113565, -0.0129259, -0.0113500, -0.0005980, 0.0009275
8: -0.0135044, -0.0119442, -0.0136797, -0.0119442, -0.0006131, 0.0007711
9: 0.0018022, 0.0025592, 0.0018025, 0.0026273, -0.0003483, 0.0002884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000619
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000619
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0000253, 0.0008208, -0.0000493, 0.0006848, -0.0003099, 0.0004713
1: -0.0034877, -0.0033264, -0.0034872, -0.0033295, -0.0000544, 0.0000580
2: 0.0149045, 0.0159725, 0.0148733, 0.0158216, -0.0003823, 0.0005662
3: 1.0066880, 1.0069630, 1.0066675, 1.0069184, -0.0001182, 0.0001848
4: -0.0042469, -0.0040827, -0.0042275, -0.0040779, -0.0000817, 0.0000572
5: 0.0039605, 0.0046068, 0.0039422, 0.0045046, -0.0002359, 0.0003576
6: -0.0026076, -0.0025656, -0.0026072, -0.0025812, -0.0000184, 0.0000338
7: -0.0129259, -0.0113567, -0.0125543, -0.0113227, -0.0010258, 0.0006197
8: -0.0136797, -0.0119651, -0.0135028, -0.0119167, -0.0008156, 0.0005858
9: 0.0018130, 0.0026273, 0.0017917, 0.0025592, -0.0002726, 0.0003654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000587
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000593
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0000253, 0.0008208, -0.0000397, 0.0006911, -0.0003020, 0.0004468
1: -0.0034877, -0.0033264, -0.0034905, -0.0033288, -0.0000521, 0.0000589
2: 0.0149045, 0.0159725, 0.0148829, 0.0158270, -0.0003750, 0.0005430
3: 1.0066880, 1.0069630, 1.0066613, 1.0069410, -0.0001114, 0.0001605
4: -0.0042469, -0.0040827, -0.0042279, -0.0040790, -0.0000796, 0.0000566
5: 0.0039605, 0.0046068, 0.0039493, 0.0045092, -0.0002302, 0.0003395
6: -0.0026076, -0.0025656, -0.0026082, -0.0025774, -0.0000174, 0.0000300
7: -0.0129259, -0.0113567, -0.0125833, -0.0113502, -0.0009370, 0.0005887
8: -0.0136797, -0.0119651, -0.0135044, -0.0119240, -0.0008027, 0.0005825
9: 0.0018130, 0.0026273, 0.0017924, 0.0025592, -0.0002723, 0.0003644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000587
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000593
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0000253, 0.0008208, -0.0000416, 0.0008138, -0.0003178, 0.0003446
1: -0.0034877, -0.0033264, -0.0034866, -0.0033270, -0.0000555, 0.0000560
2: 0.0149045, 0.0159725, 0.0148855, 0.0159657, -0.0003878, 0.0004194
3: 1.0066880, 1.0069630, 1.0066999, 1.0069375, -0.0001209, 0.0001351
4: -0.0042469, -0.0040827, -0.0042465, -0.0040800, -0.0000619, 0.0000574
5: 0.0039605, 0.0046068, 0.0039482, 0.0046017, -0.0002416, 0.0002619
6: -0.0026076, -0.0025656, -0.0026069, -0.0025688, -0.0000210, 0.0000238
7: -0.0129259, -0.0113567, -0.0128889, -0.0113301, -0.0007283, 0.0006658
8: -0.0136797, -0.0119651, -0.0136772, -0.0119382, -0.0006293, 0.0005865
9: 0.0018130, 0.0026273, 0.0018020, 0.0026271, -0.0002743, 0.0002912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000600
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000619
time: 0.64 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0000253, 0.0008208, -0.0000332, 0.0008208, -0.0003059, 0.0003175
1: -0.0034877, -0.0033264, -0.0034898, -0.0033264, -0.0000527, 0.0000564
2: 0.0149045, 0.0159725, 0.0148935, 0.0159725, -0.0003782, 0.0003947
3: 1.0066880, 1.0069630, 1.0066880, 1.0069662, -0.0001120, 0.0001074
4: -0.0042469, -0.0040827, -0.0042469, -0.0040808, -0.0000597, 0.0000569
5: 0.0039605, 0.0046068, 0.0039544, 0.0046068, -0.0002330, 0.0002420
6: -0.0026076, -0.0025656, -0.0026076, -0.0025654, -0.0000187, 0.0000185
7: -0.0129259, -0.0113567, -0.0129259, -0.0113500, -0.0006194, 0.0006100
8: -0.0136797, -0.0119651, -0.0136797, -0.0119442, -0.0006178, 0.0005861
9: 0.0018130, 0.0026273, 0.0018025, 0.0026273, -0.0002745, 0.0002905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: B, layer: 3, pos: 107
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000600
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000619
time: 0.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.44 seconds
IS_A2_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000619
IS_A2_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000619
IS_A2_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000619
IS_A2_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000619
IS_A2_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000619
IS_A2_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000619
IS_A2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000619
IS_A2_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000536, upper bound: 0.0000619
IS_A2_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000587
IS_A2_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000593
IS_A2_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000587
IS_A2_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000576, upper bound: 0.0000593
IS_A2_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000600
IS_A2_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000619
IS_A2_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000600
IS_A2_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.44
Output dim: 3, lower bound: -0.0000545, upper bound: 0.0000619

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.84 + 82.92 = 85.76 seconds
