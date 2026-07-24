## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000418


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0011228, -0.0005643, -0.0011228, -0.0005643, -0.0003376, 0.0003376)
1: (-0.0071600, -0.0057427, -0.0071600, -0.0057427, -0.0008566, 0.0008566)
2: (0.0305880, 0.0314672, 0.0305880, 0.0314672, -0.0005314, 0.0005314)
3: (0.0008052, 0.0024471, 0.0008052, 0.0024471, -0.0009924, 0.0009924)
4: (-0.0061759, -0.0047343, -0.0061759, -0.0047343, -0.0008713, 0.0008713)
5: (0.0113989, 0.0119450, 0.0113989, 0.0119450, -0.0003300, 0.0003300)
6: (0.0014100, 0.0034937, 0.0014100, 0.0034937, -0.0012594, 0.0012594)
7: (0.9790459, 0.9805040, 0.9790459, 0.9805040, -0.0008813, 0.0008813)
8: (-0.0090303, -0.0074670, -0.0090303, -0.0074670, -0.0009449, 0.0009449)
9: (-0.0000672, 0.0009654, -0.0000672, 0.0009654, -0.0006241, 0.0006241)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.13 + 1.38 = 3.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0005595, upper bound: 0.0005595

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 91
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 91

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005496, upper bound: 0.0005126
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005496, upper bound: 0.0005497
time: 0.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.22 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 7, lower bound: -0.0005496, upper bound: 0.0005126
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 7, lower bound: -0.0005496, upper bound: 0.0005497

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0010978, -0.0005648, -0.0011141, -0.0005645, -0.0003113, 0.0003278
1: -0.0070964, -0.0057439, -0.0071377, -0.0057431, -0.0007899, 0.0008317
2: 0.0306274, 0.0314665, 0.0306018, 0.0314670, -0.0004900, 0.0005160
3: 0.0008066, 0.0023734, 0.0008057, 0.0024213, -0.0009635, 0.0009150
4: -0.0061113, -0.0047355, -0.0061533, -0.0047347, -0.0008034, 0.0008460
5: 0.0114234, 0.0119445, 0.0114075, 0.0119448, -0.0003043, 0.0003204
6: 0.0014118, 0.0034003, 0.0014106, 0.0034611, -0.0012228, 0.0011613
7: 0.9790472, 0.9804386, 0.9790464, 0.9804811, -0.0008557, 0.0008126
8: -0.0090290, -0.0075371, -0.0090298, -0.0074915, -0.0009174, 0.0008713
9: -0.0000209, 0.0009645, -0.0000510, 0.0009651, -0.0005755, 0.0006060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005375, upper bound: 0.0004921
time: 0.52 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005375, upper bound: 0.0005012
time: 0.52 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0011124, -0.0005431, -0.0011183, -0.0005645, -0.0003202, 0.0003505
1: -0.0071333, -0.0056888, -0.0071483, -0.0057431, -0.0008125, 0.0008896
2: 0.0306045, 0.0315007, 0.0305952, 0.0314670, -0.0005041, 0.0005519
3: 0.0007427, 0.0024162, 0.0008057, 0.0024336, -0.0010305, 0.0009412
4: -0.0061488, -0.0046795, -0.0061641, -0.0047347, -0.0008264, 0.0009048
5: 0.0114092, 0.0119657, 0.0114034, 0.0119448, -0.0003130, 0.0003427
6: 0.0013308, 0.0034546, 0.0014106, 0.0034767, -0.0013079, 0.0011945
7: 0.9789905, 0.9804766, 0.9790463, 0.9804921, -0.0009152, 0.0008359
8: -0.0090898, -0.0074964, -0.0090299, -0.0074798, -0.0009812, 0.0008962
9: -0.0000478, 0.0010047, -0.0000588, 0.0009651, -0.0005920, 0.0006481

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005375, upper bound: 0.0005228
time: 0.51 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005375, upper bound: 0.0005375
time: 0.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0005375, upper bound: 0.0004921
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0005375, upper bound: 0.0005012
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0005375, upper bound: 0.0005228
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.90
Output dim: 7, lower bound: -0.0005375, upper bound: 0.0005375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010968, -0.0005722, -0.0011114, -0.0005847, -0.0002888, 0.0003180
1: -0.0070939, -0.0057626, -0.0071309, -0.0057944, -0.0007328, 0.0008071
2: 0.0306290, 0.0314549, 0.0306060, 0.0314351, -0.0004546, 0.0005007
3: 0.0008282, 0.0023705, 0.0008651, 0.0024134, -0.0009350, 0.0008489
4: -0.0061087, -0.0047545, -0.0061464, -0.0047869, -0.0007454, 0.0008209
5: 0.0114244, 0.0119373, 0.0114101, 0.0119250, -0.0002823, 0.0003110
6: 0.0014393, 0.0033966, 0.0014861, 0.0034511, -0.0011866, 0.0010774
7: 0.9790664, 0.9804361, 0.9790992, 0.9804742, -0.0008303, 0.0007539
8: -0.0090083, -0.0075399, -0.0089732, -0.0074990, -0.0008902, 0.0008083
9: -0.0000191, 0.0009509, -0.0000461, 0.0009277, -0.0005339, 0.0005881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0004786
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0004779
time: 0.52 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010972, -0.0005699, -0.0011253, -0.0005756, -0.0002978, 0.0003447
1: -0.0070948, -0.0057569, -0.0071662, -0.0057713, -0.0007557, 0.0008748
2: 0.0306284, 0.0314584, 0.0305841, 0.0314495, -0.0004689, 0.0005427
3: 0.0008217, 0.0023716, 0.0008384, 0.0024543, -0.0010134, 0.0008755
4: -0.0061096, -0.0047488, -0.0061823, -0.0047634, -0.0007687, 0.0008898
5: 0.0114240, 0.0119395, 0.0113965, 0.0119339, -0.0002912, 0.0003370
6: 0.0014309, 0.0033979, 0.0014521, 0.0035030, -0.0012862, 0.0011111
7: 0.9790606, 0.9804369, 0.9790754, 0.9805105, -0.0009000, 0.0007775
8: -0.0090146, -0.0075389, -0.0089987, -0.0074601, -0.0009650, 0.0008336
9: -0.0000197, 0.0009551, -0.0000718, 0.0009445, -0.0005506, 0.0006374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0004875
time: 0.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0004876
time: 0.52 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011115, -0.0005501, -0.0011156, -0.0005847, -0.0002977, 0.0003416
1: -0.0071311, -0.0057065, -0.0071417, -0.0057944, -0.0007555, 0.0008667
2: 0.0306059, 0.0314897, 0.0305993, 0.0314351, -0.0004687, 0.0005377
3: 0.0007633, 0.0024136, 0.0008652, 0.0024259, -0.0010041, 0.0008752
4: -0.0061465, -0.0046975, -0.0061573, -0.0047869, -0.0007685, 0.0008816
5: 0.0114100, 0.0119589, 0.0114060, 0.0119250, -0.0002911, 0.0003339
6: 0.0013568, 0.0034513, 0.0014861, 0.0034669, -0.0012743, 0.0011108
7: 0.9790087, 0.9804743, 0.9790992, 0.9804853, -0.0008917, 0.0007773
8: -0.0090702, -0.0074988, -0.0089732, -0.0074871, -0.0009560, 0.0008333
9: -0.0000462, 0.0009918, -0.0000539, 0.0009277, -0.0005505, 0.0006315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0005073
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0005123
time: 0.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011117, -0.0005485, -0.0011300, -0.0005757, -0.0003068, 0.0003655
1: -0.0071318, -0.0057025, -0.0071780, -0.0057714, -0.0007786, 0.0009276
2: 0.0306054, 0.0314922, 0.0305767, 0.0314494, -0.0004831, 0.0005755
3: 0.0007587, 0.0024144, 0.0008384, 0.0024680, -0.0010746, 0.0009020
4: -0.0061472, -0.0046934, -0.0061943, -0.0047635, -0.0007920, 0.0009435
5: 0.0114098, 0.0119604, 0.0113919, 0.0119339, -0.0003000, 0.0003574
6: 0.0013510, 0.0034523, 0.0014522, 0.0035203, -0.0013638, 0.0011447
7: 0.9790046, 0.9804750, 0.9790754, 0.9805226, -0.0009543, 0.0008010
8: -0.0090746, -0.0074981, -0.0089986, -0.0074470, -0.0010232, 0.0008588
9: -0.0000467, 0.0009947, -0.0000804, 0.0009445, -0.0005673, 0.0006759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0005207
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0005263
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0004786
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0004779
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0004875
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0004876
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0005073
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0005123
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0005207
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 7, lower bound: -0.0005263, upper bound: 0.0005263

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010839, -0.0005723, -0.0011067, -0.0005848, -0.0002781, 0.0003157
1: -0.0070612, -0.0057628, -0.0071190, -0.0057946, -0.0007058, 0.0008012
2: 0.0306492, 0.0314548, 0.0306134, 0.0314350, -0.0004379, 0.0004971
3: 0.0008285, 0.0023327, 0.0008654, 0.0023995, -0.0009282, 0.0008176
4: -0.0060755, -0.0047547, -0.0061342, -0.0047871, -0.0007179, 0.0008150
5: 0.0114370, 0.0119372, 0.0114147, 0.0119249, -0.0002719, 0.0003087
6: 0.0014396, 0.0033486, 0.0014864, 0.0034335, -0.0011780, 0.0010377
7: 0.9790666, 0.9804025, 0.9790993, 0.9804618, -0.0008243, 0.0007261
8: -0.0090081, -0.0075759, -0.0089730, -0.0075122, -0.0008838, 0.0007785
9: 0.0000047, 0.0009508, -0.0000374, 0.0009276, -0.0005142, 0.0005838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004786
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004785
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010881, -0.0005724, -0.0011083, -0.0005848, -0.0002832, 0.0003144
1: -0.0070718, -0.0057630, -0.0071231, -0.0057945, -0.0007186, 0.0007977
2: 0.0306427, 0.0314546, 0.0306108, 0.0314351, -0.0004458, 0.0004949
3: 0.0008287, 0.0023449, 0.0008653, 0.0024043, -0.0009241, 0.0008324
4: -0.0060862, -0.0047550, -0.0061384, -0.0047871, -0.0007309, 0.0008114
5: 0.0114329, 0.0119371, 0.0114131, 0.0119250, -0.0002768, 0.0003074
6: 0.0014399, 0.0033641, 0.0014863, 0.0034395, -0.0011729, 0.0010565
7: 0.9790668, 0.9804133, 0.9790993, 0.9804661, -0.0008207, 0.0007393
8: -0.0090079, -0.0075643, -0.0089731, -0.0075077, -0.0008799, 0.0007926
9: -0.0000030, 0.0009506, -0.0000404, 0.0009276, -0.0005236, 0.0005812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004779
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004779
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010843, -0.0005704, -0.0011210, -0.0005757, -0.0002871, 0.0003434
1: -0.0070622, -0.0057580, -0.0071553, -0.0057716, -0.0007285, 0.0008715
2: 0.0306486, 0.0314578, 0.0305908, 0.0314493, -0.0004520, 0.0005407
3: 0.0008229, 0.0023338, 0.0008387, 0.0024417, -0.0010096, 0.0008439
4: -0.0060765, -0.0047499, -0.0061712, -0.0047637, -0.0007410, 0.0008865
5: 0.0114366, 0.0119391, 0.0114007, 0.0119338, -0.0002807, 0.0003358
6: 0.0014325, 0.0033500, 0.0014525, 0.0034869, -0.0012813, 0.0010710
7: 0.9790617, 0.9804035, 0.9790757, 0.9804993, -0.0008966, 0.0007495
8: -0.0090134, -0.0075748, -0.0089984, -0.0074721, -0.0009613, 0.0008035
9: 0.0000040, 0.0009543, -0.0000639, 0.0009444, -0.0005308, 0.0006350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004875
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004875
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010884, -0.0005701, -0.0011221, -0.0005757, -0.0002919, 0.0003406
1: -0.0070726, -0.0057574, -0.0071580, -0.0057715, -0.0007407, 0.0008644
2: 0.0306421, 0.0314581, 0.0305891, 0.0314493, -0.0004596, 0.0005363
3: 0.0008223, 0.0023459, 0.0008386, 0.0024448, -0.0010013, 0.0008581
4: -0.0060871, -0.0047493, -0.0061740, -0.0047636, -0.0007534, 0.0008792
5: 0.0114326, 0.0119393, 0.0113997, 0.0119338, -0.0002854, 0.0003330
6: 0.0014317, 0.0033654, 0.0014524, 0.0034909, -0.0012708, 0.0010890
7: 0.9790611, 0.9804142, 0.9790756, 0.9805021, -0.0008893, 0.0007621
8: -0.0090141, -0.0075633, -0.0089985, -0.0074691, -0.0009534, 0.0008170
9: -0.0000036, 0.0009547, -0.0000658, 0.0009444, -0.0005397, 0.0006298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004876
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004876
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010982, -0.0005527, -0.0011112, -0.0005848, -0.0002860, 0.0003364
1: -0.0070974, -0.0057131, -0.0071303, -0.0057947, -0.0007258, 0.0008536
2: 0.0306268, 0.0314856, 0.0306063, 0.0314350, -0.0004503, 0.0005296
3: 0.0007709, 0.0023746, 0.0008654, 0.0024127, -0.0009888, 0.0008408
4: -0.0061123, -0.0047042, -0.0061458, -0.0047872, -0.0007382, 0.0008682
5: 0.0114230, 0.0119564, 0.0114103, 0.0119249, -0.0002796, 0.0003289
6: 0.0013665, 0.0034018, 0.0014864, 0.0034502, -0.0012549, 0.0010670
7: 0.9790154, 0.9804397, 0.9790993, 0.9804736, -0.0008781, 0.0007467
8: -0.0090630, -0.0075360, -0.0089730, -0.0074997, -0.0009415, 0.0008005
9: -0.0000217, 0.0009870, -0.0000457, 0.0009275, -0.0005288, 0.0006219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004875, upper bound: 0.0005073
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004875, upper bound: 0.0005073
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0005502, -0.0011128, -0.0005848, -0.0002919, 0.0003377
1: -0.0071125, -0.0057068, -0.0071345, -0.0057946, -0.0007408, 0.0008571
2: 0.0306174, 0.0314895, 0.0306038, 0.0314351, -0.0004596, 0.0005317
3: 0.0007637, 0.0023920, 0.0008653, 0.0024175, -0.0009929, 0.0008582
4: -0.0061276, -0.0046978, -0.0061500, -0.0047871, -0.0007535, 0.0008718
5: 0.0114172, 0.0119588, 0.0114087, 0.0119250, -0.0002854, 0.0003302
6: 0.0013573, 0.0034239, 0.0014863, 0.0034563, -0.0012601, 0.0010891
7: 0.9790091, 0.9804552, 0.9790993, 0.9804779, -0.0008818, 0.0007621
8: -0.0090698, -0.0075194, -0.0089731, -0.0074951, -0.0009454, 0.0008171
9: -0.0000326, 0.0009915, -0.0000487, 0.0009276, -0.0005398, 0.0006245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0005123
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0005123
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010984, -0.0005509, -0.0011256, -0.0005758, -0.0002950, 0.0003592
1: -0.0070980, -0.0057085, -0.0071670, -0.0057716, -0.0007487, 0.0009116
2: 0.0306264, 0.0314885, 0.0305836, 0.0314493, -0.0004645, 0.0005656
3: 0.0007656, 0.0023752, 0.0008387, 0.0024552, -0.0010561, 0.0008673
4: -0.0061128, -0.0046995, -0.0061831, -0.0047637, -0.0007616, 0.0009273
5: 0.0114228, 0.0119581, 0.0113962, 0.0119338, -0.0002885, 0.0003512
6: 0.0013598, 0.0034026, 0.0014526, 0.0035041, -0.0013403, 0.0011008
7: 0.9790108, 0.9804403, 0.9790757, 0.9805113, -0.0009379, 0.0007703
8: -0.0090680, -0.0075354, -0.0089984, -0.0074592, -0.0010055, 0.0008258
9: -0.0000221, 0.0009903, -0.0000724, 0.0009443, -0.0005455, 0.0006642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004875, upper bound: 0.0005206
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004875, upper bound: 0.0005207
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0011044, -0.0005487, -0.0011265, -0.0005757, -0.0003007, 0.0003622
1: -0.0071131, -0.0057029, -0.0071693, -0.0057716, -0.0007630, 0.0009190
2: 0.0306170, 0.0314919, 0.0305821, 0.0314493, -0.0004734, 0.0005702
3: 0.0007591, 0.0023928, 0.0008387, 0.0024579, -0.0010647, 0.0008839
4: -0.0061283, -0.0046938, -0.0061854, -0.0047637, -0.0007761, 0.0009348
5: 0.0114170, 0.0119603, 0.0113953, 0.0119338, -0.0002940, 0.0003541
6: 0.0013516, 0.0034249, 0.0014525, 0.0035075, -0.0013512, 0.0011218
7: 0.9790050, 0.9804558, 0.9790757, 0.9805137, -0.0009455, 0.0007850
8: -0.0090742, -0.0075187, -0.0089984, -0.0074566, -0.0010137, 0.0008416
9: -0.0000331, 0.0009944, -0.0000741, 0.0009444, -0.0005559, 0.0006696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 91
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 91

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0005263
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0005263
time: 0.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.06 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004786
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004785
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004779
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004779
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004875
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004875
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004876
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0005071, upper bound: 0.0004876
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0004875, upper bound: 0.0005073
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0004875, upper bound: 0.0005073
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0005123
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0005123
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0004875, upper bound: 0.0005206
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0004875, upper bound: 0.0005207
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0005263
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0005263

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010839, -0.0005723, -0.0010903, -0.0005851, -0.0002779, 0.0002991
1: -0.0070612, -0.0057628, -0.0070773, -0.0057953, -0.0007051, 0.0007591
2: 0.0306492, 0.0314548, 0.0306392, 0.0314346, -0.0004374, 0.0004709
3: 0.0008285, 0.0023327, 0.0008662, 0.0023513, -0.0008793, 0.0008168
4: -0.0060755, -0.0047547, -0.0060918, -0.0047878, -0.0007172, 0.0007721
5: 0.0114370, 0.0119372, 0.0114308, 0.0119247, -0.0002717, 0.0002924
6: 0.0014396, 0.0033486, 0.0014874, 0.0033722, -0.0011160, 0.0010367
7: 0.9790666, 0.9804025, 0.9791002, 0.9804190, -0.0007809, 0.0007254
8: -0.0090081, -0.0075759, -0.0089722, -0.0075582, -0.0008373, 0.0007777
9: 0.0000047, 0.0009508, -0.0000070, 0.0009271, -0.0005137, 0.0005531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004637
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004670
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010839, -0.0005723, -0.0011051, -0.0005625, -0.0002973, 0.0003144
1: -0.0070612, -0.0057628, -0.0071149, -0.0057379, -0.0007545, 0.0007979
2: 0.0306492, 0.0314548, 0.0306159, 0.0314702, -0.0004681, 0.0004950
3: 0.0008285, 0.0023327, 0.0007997, 0.0023949, -0.0009243, 0.0008740
4: -0.0060755, -0.0047547, -0.0061301, -0.0047294, -0.0007674, 0.0008116
5: 0.0114370, 0.0119372, 0.0114163, 0.0119468, -0.0002907, 0.0003074
6: 0.0014396, 0.0033486, 0.0014030, 0.0034275, -0.0011731, 0.0011093
7: 0.9790666, 0.9804025, 0.9790410, 0.9804577, -0.0008208, 0.0007762
8: -0.0090081, -0.0075759, -0.0090356, -0.0075167, -0.0008801, 0.0008322
9: 0.0000047, 0.0009508, -0.0000344, 0.0009689, -0.0005497, 0.0005813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004637
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004670
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010881, -0.0005724, -0.0010919, -0.0005851, -0.0002828, 0.0002976
1: -0.0070718, -0.0057630, -0.0070814, -0.0057952, -0.0007177, 0.0007551
2: 0.0306427, 0.0314546, 0.0306367, 0.0314346, -0.0004452, 0.0004685
3: 0.0008287, 0.0023449, 0.0008661, 0.0023560, -0.0008748, 0.0008314
4: -0.0060862, -0.0047550, -0.0060960, -0.0047878, -0.0007300, 0.0007681
5: 0.0114329, 0.0119371, 0.0114292, 0.0119247, -0.0002765, 0.0002909
6: 0.0014399, 0.0033641, 0.0014873, 0.0033782, -0.0011102, 0.0010551
7: 0.9790668, 0.9804133, 0.9791000, 0.9804233, -0.0007769, 0.0007383
8: -0.0090079, -0.0075643, -0.0089723, -0.0075537, -0.0008329, 0.0007916
9: -0.0000030, 0.0009506, -0.0000100, 0.0009271, -0.0005229, 0.0005502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004630
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004668
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010881, -0.0005724, -0.0011072, -0.0005624, -0.0003002, 0.0003131
1: -0.0070718, -0.0057630, -0.0071203, -0.0057378, -0.0007619, 0.0007945
2: 0.0306427, 0.0314546, 0.0306126, 0.0314703, -0.0004727, 0.0004929
3: 0.0008287, 0.0023449, 0.0007996, 0.0024011, -0.0009204, 0.0008826
4: -0.0060862, -0.0047550, -0.0061355, -0.0047293, -0.0007750, 0.0008082
5: 0.0114329, 0.0119371, 0.0114142, 0.0119468, -0.0002935, 0.0003061
6: 0.0014399, 0.0033641, 0.0014029, 0.0034354, -0.0011681, 0.0011201
7: 0.9790668, 0.9804133, 0.9790409, 0.9804632, -0.0008174, 0.0007838
8: -0.0090079, -0.0075643, -0.0090357, -0.0075108, -0.0008764, 0.0008404
9: -0.0000030, 0.0009506, -0.0000383, 0.0009690, -0.0005551, 0.0005789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004630
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004667
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010843, -0.0005704, -0.0011048, -0.0005761, -0.0002867, 0.0003281
1: -0.0070622, -0.0057580, -0.0071143, -0.0057725, -0.0007274, 0.0008325
2: 0.0306486, 0.0314578, 0.0306163, 0.0314487, -0.0004513, 0.0005165
3: 0.0008229, 0.0023338, 0.0008397, 0.0023941, -0.0009644, 0.0008427
4: -0.0060765, -0.0047499, -0.0061294, -0.0047646, -0.0007399, 0.0008468
5: 0.0114366, 0.0119391, 0.0114165, 0.0119335, -0.0002803, 0.0003208
6: 0.0014325, 0.0033500, 0.0014539, 0.0034266, -0.0012240, 0.0010695
7: 0.9790617, 0.9804035, 0.9790766, 0.9804571, -0.0008565, 0.0007484
8: -0.0090134, -0.0075748, -0.0089974, -0.0075174, -0.0009183, 0.0008024
9: 0.0000040, 0.0009543, -0.0000339, 0.0009437, -0.0005300, 0.0006066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004721
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004770
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010843, -0.0005704, -0.0011192, -0.0005549, -0.0003052, 0.0003383
1: -0.0070622, -0.0057580, -0.0071507, -0.0057186, -0.0007745, 0.0008584
2: 0.0306486, 0.0314578, 0.0305937, 0.0314822, -0.0004805, 0.0005326
3: 0.0008229, 0.0023338, 0.0007774, 0.0024364, -0.0009944, 0.0008973
4: -0.0060765, -0.0047499, -0.0061665, -0.0047098, -0.0007878, 0.0008732
5: 0.0114366, 0.0119391, 0.0114025, 0.0119542, -0.0002984, 0.0003307
6: 0.0014325, 0.0033500, 0.0013747, 0.0034802, -0.0012621, 0.0011387
7: 0.9790617, 0.9804035, 0.9790213, 0.9804946, -0.0008831, 0.0007968
8: -0.0090134, -0.0075748, -0.0090568, -0.0074772, -0.0009469, 0.0008543
9: 0.0000040, 0.0009543, -0.0000605, 0.0009829, -0.0005643, 0.0006255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004721
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004770
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010884, -0.0005701, -0.0011059, -0.0005761, -0.0002914, 0.0003246
1: -0.0070726, -0.0057574, -0.0071169, -0.0057724, -0.0007395, 0.0008238
2: 0.0306421, 0.0314581, 0.0306147, 0.0314488, -0.0004588, 0.0005111
3: 0.0008223, 0.0023459, 0.0008397, 0.0023972, -0.0009543, 0.0008567
4: -0.0060871, -0.0047493, -0.0061321, -0.0047646, -0.0007522, 0.0008379
5: 0.0114326, 0.0119393, 0.0114155, 0.0119335, -0.0002849, 0.0003174
6: 0.0014317, 0.0033654, 0.0014538, 0.0034304, -0.0012112, 0.0010872
7: 0.9790611, 0.9804142, 0.9790766, 0.9804597, -0.0008475, 0.0007608
8: -0.0090141, -0.0075633, -0.0089975, -0.0075145, -0.0009087, 0.0008157
9: -0.0000036, 0.0009547, -0.0000359, 0.0009437, -0.0005388, 0.0006002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004723
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004766
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010884, -0.0005701, -0.0011201, -0.0005549, -0.0003068, 0.0003373
1: -0.0070726, -0.0057574, -0.0071531, -0.0057186, -0.0007786, 0.0008559
2: 0.0306421, 0.0314581, 0.0305922, 0.0314822, -0.0004830, 0.0005310
3: 0.0008223, 0.0023459, 0.0007773, 0.0024391, -0.0009915, 0.0009020
4: -0.0060871, -0.0047493, -0.0061689, -0.0047098, -0.0007920, 0.0008706
5: 0.0114326, 0.0119393, 0.0114016, 0.0119542, -0.0003000, 0.0003297
6: 0.0014317, 0.0033654, 0.0013746, 0.0034836, -0.0012583, 0.0011447
7: 0.9790611, 0.9804142, 0.9790212, 0.9804970, -0.0008805, 0.0008010
8: -0.0090141, -0.0075633, -0.0090569, -0.0074746, -0.0009440, 0.0008588
9: -0.0000036, 0.0009547, -0.0000622, 0.0009830, -0.0005673, 0.0006236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004722
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004766
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010982, -0.0005527, -0.0010903, -0.0005851, -0.0002919, 0.0003151
1: -0.0070974, -0.0057131, -0.0070773, -0.0057953, -0.0007407, 0.0007995
2: 0.0306268, 0.0314856, 0.0306392, 0.0314346, -0.0004595, 0.0004960
3: 0.0007709, 0.0023746, 0.0008662, 0.0023513, -0.0009262, 0.0008581
4: -0.0061123, -0.0047042, -0.0060918, -0.0047878, -0.0007534, 0.0008132
5: 0.0114230, 0.0119564, 0.0114308, 0.0119247, -0.0002854, 0.0003080
6: 0.0013665, 0.0034018, 0.0014874, 0.0033722, -0.0011755, 0.0010890
7: 0.9790154, 0.9804397, 0.9791002, 0.9804190, -0.0008225, 0.0007620
8: -0.0090630, -0.0075360, -0.0089722, -0.0075582, -0.0008819, 0.0008170
9: -0.0000217, 0.0009870, -0.0000070, 0.0009271, -0.0005397, 0.0005825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004875
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0004982
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010982, -0.0005527, -0.0011051, -0.0005625, -0.0002942, 0.0003147
1: -0.0070974, -0.0057131, -0.0071149, -0.0057379, -0.0007467, 0.0007985
2: 0.0306268, 0.0314856, 0.0306159, 0.0314702, -0.0004632, 0.0004954
3: 0.0007709, 0.0023746, 0.0007997, 0.0023949, -0.0009250, 0.0008650
4: -0.0061123, -0.0047042, -0.0061301, -0.0047294, -0.0007595, 0.0008122
5: 0.0114230, 0.0119564, 0.0114163, 0.0119468, -0.0002877, 0.0003076
6: 0.0013665, 0.0034018, 0.0014030, 0.0034275, -0.0011740, 0.0010977
7: 0.9790154, 0.9804397, 0.9790410, 0.9804577, -0.0008215, 0.0007681
8: -0.0090630, -0.0075360, -0.0090356, -0.0075167, -0.0008808, 0.0008236
9: -0.0000217, 0.0009870, -0.0000344, 0.0009689, -0.0005440, 0.0005818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004876
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0004982
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0005502, -0.0010919, -0.0005851, -0.0002969, 0.0003165
1: -0.0071125, -0.0057068, -0.0070814, -0.0057952, -0.0007534, 0.0008032
2: 0.0306174, 0.0314895, 0.0306367, 0.0314346, -0.0004674, 0.0004983
3: 0.0007637, 0.0023920, 0.0008661, 0.0023560, -0.0009305, 0.0008728
4: -0.0061276, -0.0046978, -0.0060960, -0.0047878, -0.0007663, 0.0008170
5: 0.0114172, 0.0119588, 0.0114292, 0.0119247, -0.0002903, 0.0003095
6: 0.0013573, 0.0034239, 0.0014873, 0.0033782, -0.0011809, 0.0011076
7: 0.9790091, 0.9804552, 0.9791000, 0.9804233, -0.0008263, 0.0007751
8: -0.0090698, -0.0075194, -0.0089723, -0.0075537, -0.0008860, 0.0008310
9: -0.0000326, 0.0009915, -0.0000100, 0.0009271, -0.0005489, 0.0005852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0004923
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0005029
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011041, -0.0005502, -0.0011072, -0.0005624, -0.0002991, 0.0003148
1: -0.0071125, -0.0057068, -0.0071203, -0.0057378, -0.0007591, 0.0007988
2: 0.0306174, 0.0314895, 0.0306126, 0.0314703, -0.0004710, 0.0004956
3: 0.0007637, 0.0023920, 0.0007996, 0.0024011, -0.0009253, 0.0008794
4: -0.0061276, -0.0046978, -0.0061355, -0.0047293, -0.0007722, 0.0008125
5: 0.0114172, 0.0119588, 0.0114142, 0.0119468, -0.0002925, 0.0003077
6: 0.0013573, 0.0034239, 0.0014029, 0.0034354, -0.0011744, 0.0011161
7: 0.9790091, 0.9804552, 0.9790409, 0.9804632, -0.0008218, 0.0007810
8: -0.0090698, -0.0075194, -0.0090357, -0.0075108, -0.0008811, 0.0008373
9: -0.0000326, 0.0009915, -0.0000383, 0.0009690, -0.0005531, 0.0005820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0004926
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0005029
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010984, -0.0005509, -0.0011048, -0.0005761, -0.0003007, 0.0003413
1: -0.0070980, -0.0057085, -0.0071143, -0.0057725, -0.0007632, 0.0008661
2: 0.0306264, 0.0314885, 0.0306163, 0.0314487, -0.0004735, 0.0005374
3: 0.0007656, 0.0023752, 0.0008397, 0.0023941, -0.0010034, 0.0008841
4: -0.0061128, -0.0046995, -0.0061294, -0.0047646, -0.0007763, 0.0008810
5: 0.0114228, 0.0119581, 0.0114165, 0.0119335, -0.0002940, 0.0003337
6: 0.0013598, 0.0034026, 0.0014539, 0.0034266, -0.0012734, 0.0011220
7: 0.9790108, 0.9804403, 0.9790766, 0.9804571, -0.0008911, 0.0007851
8: -0.0090680, -0.0075354, -0.0089974, -0.0075174, -0.0009554, 0.0008418
9: -0.0000221, 0.0009903, -0.0000339, 0.0009437, -0.0005560, 0.0006311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004998
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0005120
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010984, -0.0005509, -0.0011192, -0.0005549, -0.0003026, 0.0003397
1: -0.0070980, -0.0057085, -0.0071507, -0.0057186, -0.0007680, 0.0008621
2: 0.0306264, 0.0314885, 0.0305937, 0.0314822, -0.0004764, 0.0005348
3: 0.0007656, 0.0023752, 0.0007774, 0.0024364, -0.0009987, 0.0008896
4: -0.0061128, -0.0046995, -0.0061665, -0.0047098, -0.0007811, 0.0008769
5: 0.0114228, 0.0119581, 0.0114025, 0.0119542, -0.0002959, 0.0003321
6: 0.0013598, 0.0034026, 0.0013747, 0.0034802, -0.0012675, 0.0011291
7: 0.9790108, 0.9804403, 0.9790213, 0.9804946, -0.0008869, 0.0007901
8: -0.0090680, -0.0075354, -0.0090568, -0.0074772, -0.0009509, 0.0008471
9: -0.0000221, 0.0009903, -0.0000605, 0.0009829, -0.0005595, 0.0006281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004999
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0005121
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0011044, -0.0005487, -0.0011059, -0.0005761, -0.0003055, 0.0003420
1: -0.0071131, -0.0057029, -0.0071169, -0.0057724, -0.0007754, 0.0008679
2: 0.0306170, 0.0314919, 0.0306147, 0.0314488, -0.0004810, 0.0005384
3: 0.0007591, 0.0023928, 0.0008397, 0.0023972, -0.0010054, 0.0008982
4: -0.0061283, -0.0046938, -0.0061321, -0.0047646, -0.0007887, 0.0008828
5: 0.0114170, 0.0119603, 0.0114155, 0.0119335, -0.0002987, 0.0003344
6: 0.0013516, 0.0034249, 0.0014538, 0.0034304, -0.0012760, 0.0011400
7: 0.9790050, 0.9804558, 0.9790766, 0.9804597, -0.0008929, 0.0007977
8: -0.0090742, -0.0075187, -0.0089975, -0.0075145, -0.0009573, 0.0008552
9: -0.0000331, 0.0009944, -0.0000359, 0.0009437, -0.0005649, 0.0006323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0005042
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0005176
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0011044, -0.0005487, -0.0011201, -0.0005549, -0.0003068, 0.0003392
1: -0.0071131, -0.0057029, -0.0071531, -0.0057186, -0.0007786, 0.0008608
2: 0.0306170, 0.0314919, 0.0305922, 0.0314822, -0.0004830, 0.0005340
3: 0.0007591, 0.0023928, 0.0007773, 0.0024391, -0.0009971, 0.0009019
4: -0.0061283, -0.0046938, -0.0061689, -0.0047098, -0.0007919, 0.0008755
5: 0.0114170, 0.0119603, 0.0114016, 0.0119542, -0.0003000, 0.0003316
6: 0.0013516, 0.0034249, 0.0013746, 0.0034836, -0.0012655, 0.0011447
7: 0.9790050, 0.9804558, 0.9790212, 0.9804970, -0.0008855, 0.0008010
8: -0.0090742, -0.0075187, -0.0090569, -0.0074746, -0.0009494, 0.0008588
9: -0.0000331, 0.0009944, -0.0000622, 0.0009830, -0.0005673, 0.0006272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0005043
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0005177
time: 0.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.05 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004637
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004670
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004637
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004670
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004630
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004668
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004630
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004667
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004721
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004770
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004721
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004770
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004723
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004766
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004957, upper bound: 0.0004722
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004766
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004875
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0004982
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004876
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0004982
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0004923
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0005029
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0004926
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0005029
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004998
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0005120
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004999
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004765, upper bound: 0.0005121
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0005042
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0005176
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0005043
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0004766, upper bound: 0.0005177

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010704, -0.0005751, -0.0010849, -0.0005853, -0.0002602, 0.0002890
1: -0.0070269, -0.0057699, -0.0070636, -0.0057959, -0.0006603, 0.0007333
2: 0.0306705, 0.0314503, 0.0306477, 0.0314342, -0.0004096, 0.0004549
3: 0.0008368, 0.0022930, 0.0008669, 0.0023355, -0.0008495, 0.0007649
4: -0.0060406, -0.0047620, -0.0060779, -0.0047884, -0.0006716, 0.0007459
5: 0.0114502, 0.0119345, 0.0114360, 0.0119244, -0.0002544, 0.0002825
6: 0.0014501, 0.0032982, 0.0014883, 0.0033521, -0.0010781, 0.0009707
7: 0.9790741, 0.9803672, 0.9791007, 0.9804050, -0.0007544, 0.0006793
8: -0.0090002, -0.0076137, -0.0089716, -0.0075732, -0.0008088, 0.0007283
9: 0.0000297, 0.0009456, 0.0000030, 0.0009266, -0.0004811, 0.0005343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004727
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004727
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010772, -0.0005729, -0.0010882, -0.0005853, -0.0002585, 0.0002966
1: -0.0070441, -0.0057643, -0.0070719, -0.0057958, -0.0006561, 0.0007526
2: 0.0306598, 0.0314538, 0.0306426, 0.0314343, -0.0004070, 0.0004669
3: 0.0008303, 0.0023129, 0.0008667, 0.0023451, -0.0008718, 0.0007600
4: -0.0060581, -0.0047563, -0.0060863, -0.0047883, -0.0006673, 0.0007655
5: 0.0114435, 0.0119366, 0.0114328, 0.0119245, -0.0002528, 0.0002900
6: 0.0014419, 0.0033235, 0.0014881, 0.0033643, -0.0011065, 0.0009646
7: 0.9790682, 0.9803848, 0.9791006, 0.9804134, -0.0007742, 0.0006750
8: -0.0090064, -0.0075947, -0.0089717, -0.0075641, -0.0008301, 0.0007237
9: 0.0000172, 0.0009496, -0.0000031, 0.0009267, -0.0004780, 0.0005483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0004794
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0004804
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010704, -0.0005751, -0.0010991, -0.0005627, -0.0002796, 0.0003043
1: -0.0070269, -0.0057699, -0.0070996, -0.0057385, -0.0007096, 0.0007722
2: 0.0306705, 0.0314503, 0.0306254, 0.0314698, -0.0004402, 0.0004791
3: 0.0008368, 0.0022930, 0.0008003, 0.0023771, -0.0008946, 0.0008220
4: -0.0060406, -0.0047620, -0.0061145, -0.0047300, -0.0007218, 0.0007855
5: 0.0114502, 0.0119345, 0.0114222, 0.0119466, -0.0002734, 0.0002975
6: 0.0014501, 0.0032982, 0.0014038, 0.0034050, -0.0011353, 0.0010433
7: 0.9790741, 0.9803672, 0.9790417, 0.9804419, -0.0007945, 0.0007300
8: -0.0090002, -0.0076137, -0.0090349, -0.0075336, -0.0008518, 0.0007827
9: 0.0000297, 0.0009456, -0.0000232, 0.0009685, -0.0005170, 0.0005626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005122, upper bound: 0.0004637
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005122, upper bound: 0.0004637
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010772, -0.0005729, -0.0011033, -0.0005626, -0.0002817, 0.0003116
1: -0.0070441, -0.0057643, -0.0071103, -0.0057383, -0.0007148, 0.0007907
2: 0.0306598, 0.0314538, 0.0306187, 0.0314700, -0.0004435, 0.0004906
3: 0.0008303, 0.0023129, 0.0008001, 0.0023896, -0.0009160, 0.0008281
4: -0.0060581, -0.0047563, -0.0061254, -0.0047298, -0.0007271, 0.0008043
5: 0.0114435, 0.0119366, 0.0114180, 0.0119467, -0.0002754, 0.0003046
6: 0.0014419, 0.0033235, 0.0014036, 0.0034208, -0.0011625, 0.0010509
7: 0.9790682, 0.9803848, 0.9790414, 0.9804530, -0.0008135, 0.0007354
8: -0.0090064, -0.0075947, -0.0090351, -0.0075217, -0.0008722, 0.0007884
9: 0.0000172, 0.0009496, -0.0000311, 0.0009686, -0.0005208, 0.0005761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005041, upper bound: 0.0004655
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005041, upper bound: 0.0004670
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010741, -0.0005743, -0.0010865, -0.0005853, -0.0002651, 0.0002860
1: -0.0070362, -0.0057679, -0.0070678, -0.0057958, -0.0006727, 0.0007257
2: 0.0306647, 0.0314516, 0.0306452, 0.0314343, -0.0004173, 0.0004502
3: 0.0008344, 0.0023037, 0.0008668, 0.0023402, -0.0008407, 0.0007793
4: -0.0060500, -0.0047599, -0.0060821, -0.0047884, -0.0006843, 0.0007382
5: 0.0114466, 0.0119352, 0.0114344, 0.0119245, -0.0002592, 0.0002796
6: 0.0014471, 0.0033118, 0.0014882, 0.0033582, -0.0010669, 0.0009890
7: 0.9790719, 0.9803767, 0.9791006, 0.9804091, -0.0007466, 0.0006921
8: -0.0090025, -0.0076035, -0.0089717, -0.0075687, -0.0008005, 0.0007420
9: 0.0000229, 0.0009471, -0.0000001, 0.0009267, -0.0004901, 0.0005287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004757
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004758
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010815, -0.0005730, -0.0010898, -0.0005853, -0.0002649, 0.0002952
1: -0.0070550, -0.0057646, -0.0070760, -0.0057957, -0.0006723, 0.0007491
2: 0.0306531, 0.0314536, 0.0306400, 0.0314343, -0.0004171, 0.0004647
3: 0.0008306, 0.0023254, 0.0008667, 0.0023498, -0.0008677, 0.0007788
4: -0.0060691, -0.0047566, -0.0060905, -0.0047883, -0.0006838, 0.0007619
5: 0.0114394, 0.0119365, 0.0114313, 0.0119245, -0.0002590, 0.0002886
6: 0.0014423, 0.0033394, 0.0014880, 0.0033703, -0.0011013, 0.0009884
7: 0.9790685, 0.9803960, 0.9791005, 0.9804177, -0.0007706, 0.0006916
8: -0.0090061, -0.0075828, -0.0089718, -0.0075596, -0.0008262, 0.0007415
9: 0.0000093, 0.0009494, -0.0000061, 0.0009268, -0.0004898, 0.0005458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004942, upper bound: 0.0004842
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004942, upper bound: 0.0004843
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010741, -0.0005743, -0.0011014, -0.0005626, -0.0002825, 0.0003022
1: -0.0070362, -0.0057679, -0.0071056, -0.0057384, -0.0007170, 0.0007669
2: 0.0306647, 0.0314516, 0.0306217, 0.0314699, -0.0004448, 0.0004758
3: 0.0008344, 0.0023037, 0.0008002, 0.0023840, -0.0008884, 0.0008306
4: -0.0060500, -0.0047599, -0.0061206, -0.0047299, -0.0007293, 0.0007800
5: 0.0114466, 0.0119352, 0.0114199, 0.0119466, -0.0002762, 0.0002955
6: 0.0014471, 0.0033118, 0.0014037, 0.0034138, -0.0011275, 0.0010541
7: 0.9790719, 0.9803767, 0.9790416, 0.9804481, -0.0007889, 0.0007376
8: -0.0090025, -0.0076035, -0.0090350, -0.0075270, -0.0008459, 0.0007909
9: 0.0000229, 0.0009471, -0.0000276, 0.0009686, -0.0005224, 0.0005587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005106, upper bound: 0.0004628
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005106, upper bound: 0.0004630
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010815, -0.0005730, -0.0011053, -0.0005626, -0.0002830, 0.0003101
1: -0.0070550, -0.0057646, -0.0071154, -0.0057382, -0.0007180, 0.0007869
2: 0.0306531, 0.0314536, 0.0306156, 0.0314700, -0.0004455, 0.0004882
3: 0.0008306, 0.0023254, 0.0008000, 0.0023955, -0.0009116, 0.0008318
4: -0.0060691, -0.0047566, -0.0061306, -0.0047297, -0.0007304, 0.0008005
5: 0.0114394, 0.0119365, 0.0114161, 0.0119467, -0.0002766, 0.0003032
6: 0.0014423, 0.0033394, 0.0014034, 0.0034283, -0.0011570, 0.0010557
7: 0.9790685, 0.9803960, 0.9790414, 0.9804582, -0.0008096, 0.0007387
8: -0.0090061, -0.0075828, -0.0090352, -0.0075161, -0.0008680, 0.0007920
9: 0.0000093, 0.0009494, -0.0000348, 0.0009687, -0.0005232, 0.0005734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005120, upper bound: 0.0004662
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005120, upper bound: 0.0004667
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010708, -0.0005722, -0.0010996, -0.0005764, -0.0002688, 0.0003145
1: -0.0070279, -0.0057626, -0.0071010, -0.0057733, -0.0006822, 0.0007981
2: 0.0306699, 0.0314549, 0.0306246, 0.0314482, -0.0004233, 0.0004951
3: 0.0008283, 0.0022941, 0.0008407, 0.0023787, -0.0009245, 0.0007903
4: -0.0060416, -0.0047546, -0.0061159, -0.0047655, -0.0006940, 0.0008118
5: 0.0114498, 0.0119373, 0.0114216, 0.0119331, -0.0002629, 0.0003075
6: 0.0014394, 0.0032996, 0.0014551, 0.0034070, -0.0011734, 0.0010031
7: 0.9790665, 0.9803681, 0.9790776, 0.9804433, -0.0008211, 0.0007019
8: -0.0090083, -0.0076126, -0.0089965, -0.0075321, -0.0008803, 0.0007525
9: 0.0000290, 0.0009509, -0.0000242, 0.0009431, -0.0004971, 0.0005815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004861
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004861
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010776, -0.0005710, -0.0011028, -0.0005763, -0.0002659, 0.0003254
1: -0.0070450, -0.0057597, -0.0071090, -0.0057731, -0.0006747, 0.0008257
2: 0.0306592, 0.0314567, 0.0306196, 0.0314484, -0.0004186, 0.0005123
3: 0.0008249, 0.0023139, 0.0008405, 0.0023880, -0.0009566, 0.0007816
4: -0.0060590, -0.0047516, -0.0061241, -0.0047653, -0.0006863, 0.0008399
5: 0.0114432, 0.0119384, 0.0114186, 0.0119332, -0.0002599, 0.0003181
6: 0.0014350, 0.0033248, 0.0014548, 0.0034188, -0.0012140, 0.0009919
7: 0.9790635, 0.9803858, 0.9790773, 0.9804516, -0.0008495, 0.0006941
8: -0.0090115, -0.0075937, -0.0089967, -0.0075232, -0.0009108, 0.0007442
9: 0.0000165, 0.0009530, -0.0000301, 0.0009432, -0.0004916, 0.0006016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004885, upper bound: 0.0004932
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004885, upper bound: 0.0004945
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010708, -0.0005722, -0.0011131, -0.0005552, -0.0002875, 0.0003249
1: -0.0070279, -0.0057626, -0.0071353, -0.0057194, -0.0007295, 0.0008245
2: 0.0306699, 0.0314549, 0.0306033, 0.0314817, -0.0004526, 0.0005115
3: 0.0008283, 0.0022941, 0.0007783, 0.0024184, -0.0009552, 0.0008450
4: -0.0060416, -0.0047546, -0.0061508, -0.0047107, -0.0007420, 0.0008387
5: 0.0114498, 0.0119373, 0.0114084, 0.0119539, -0.0002810, 0.0003177
6: 0.0014394, 0.0032996, 0.0013759, 0.0034574, -0.0012122, 0.0010725
7: 0.9790665, 0.9803681, 0.9790220, 0.9804786, -0.0008483, 0.0007505
8: -0.0090083, -0.0076126, -0.0090559, -0.0074942, -0.0009095, 0.0008046
9: 0.0000290, 0.0009509, -0.0000492, 0.0009823, -0.0005315, 0.0006007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005122, upper bound: 0.0004721
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005122, upper bound: 0.0004721
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010776, -0.0005710, -0.0011174, -0.0005551, -0.0002884, 0.0003356
1: -0.0070450, -0.0057597, -0.0071460, -0.0057192, -0.0007320, 0.0008517
2: 0.0306592, 0.0314567, 0.0305966, 0.0314818, -0.0004541, 0.0005284
3: 0.0008249, 0.0023139, 0.0007780, 0.0024309, -0.0009867, 0.0008479
4: -0.0060590, -0.0047516, -0.0061618, -0.0047104, -0.0007445, 0.0008663
5: 0.0114432, 0.0119384, 0.0114043, 0.0119540, -0.0002820, 0.0003281
6: 0.0014350, 0.0033248, 0.0013755, 0.0034733, -0.0012522, 0.0010761
7: 0.9790635, 0.9803858, 0.9790218, 0.9804897, -0.0008762, 0.0007530
8: -0.0090115, -0.0075937, -0.0090562, -0.0074823, -0.0009395, 0.0008074
9: 0.0000165, 0.0009530, -0.0000571, 0.0009825, -0.0005333, 0.0006206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005040, upper bound: 0.0004755
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005040, upper bound: 0.0004770
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010744, -0.0005718, -0.0011007, -0.0005764, -0.0002736, 0.0003104
1: -0.0070371, -0.0057616, -0.0071037, -0.0057733, -0.0006942, 0.0007876
2: 0.0306642, 0.0314555, 0.0306229, 0.0314483, -0.0004307, 0.0004886
3: 0.0008271, 0.0023047, 0.0008406, 0.0023819, -0.0009124, 0.0008042
4: -0.0060509, -0.0047536, -0.0061187, -0.0047654, -0.0007061, 0.0008011
5: 0.0114463, 0.0119377, 0.0114206, 0.0119332, -0.0002675, 0.0003034
6: 0.0014379, 0.0033131, 0.0014550, 0.0034110, -0.0011579, 0.0010206
7: 0.9790654, 0.9803777, 0.9790775, 0.9804462, -0.0008102, 0.0007142
8: -0.0090094, -0.0076025, -0.0089965, -0.0075290, -0.0008687, 0.0007657
9: 0.0000223, 0.0009516, -0.0000262, 0.0009431, -0.0005058, 0.0005738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004886
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004886
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010818, -0.0005708, -0.0011038, -0.0005763, -0.0002724, 0.0003220
1: -0.0070558, -0.0057591, -0.0071115, -0.0057731, -0.0006912, 0.0008172
2: 0.0306525, 0.0314570, 0.0306180, 0.0314484, -0.0004288, 0.0005070
3: 0.0008243, 0.0023264, 0.0008404, 0.0023910, -0.0009467, 0.0008007
4: -0.0060700, -0.0047510, -0.0061267, -0.0047652, -0.0007031, 0.0008312
5: 0.0114390, 0.0119386, 0.0114176, 0.0119332, -0.0002663, 0.0003149
6: 0.0014342, 0.0033407, 0.0014547, 0.0034226, -0.0012015, 0.0010162
7: 0.9790629, 0.9803969, 0.9790772, 0.9804542, -0.0008407, 0.0007111
8: -0.0090121, -0.0075818, -0.0089968, -0.0075204, -0.0009014, 0.0007624
9: 0.0000086, 0.0009534, -0.0000319, 0.0009433, -0.0005036, 0.0005954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0004957
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0004969
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010744, -0.0005718, -0.0011140, -0.0005552, -0.0002890, 0.0003230
1: -0.0070371, -0.0057616, -0.0071376, -0.0057194, -0.0007335, 0.0008197
2: 0.0306642, 0.0314555, 0.0306018, 0.0314817, -0.0004551, 0.0005086
3: 0.0008271, 0.0023047, 0.0007782, 0.0024212, -0.0009496, 0.0008497
4: -0.0060509, -0.0047536, -0.0061532, -0.0047106, -0.0007461, 0.0008338
5: 0.0114463, 0.0119377, 0.0114075, 0.0119539, -0.0002826, 0.0003158
6: 0.0014379, 0.0033131, 0.0013757, 0.0034609, -0.0012052, 0.0010784
7: 0.9790654, 0.9803777, 0.9790219, 0.9804810, -0.0008433, 0.0007546
8: -0.0090094, -0.0076025, -0.0090560, -0.0074916, -0.0009042, 0.0008091
9: 0.0000223, 0.0009516, -0.0000509, 0.0009824, -0.0005344, 0.0005973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005106, upper bound: 0.0004722
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005106, upper bound: 0.0004722
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010818, -0.0005708, -0.0011183, -0.0005551, -0.0002890, 0.0003347
1: -0.0070558, -0.0057591, -0.0071484, -0.0057191, -0.0007334, 0.0008494
2: 0.0306525, 0.0314570, 0.0305951, 0.0314819, -0.0004550, 0.0005270
3: 0.0008243, 0.0023264, 0.0007779, 0.0024337, -0.0009840, 0.0008496
4: -0.0060700, -0.0047510, -0.0061641, -0.0047103, -0.0007460, 0.0008640
5: 0.0114390, 0.0119386, 0.0114034, 0.0119540, -0.0002825, 0.0003273
6: 0.0014342, 0.0033407, 0.0013754, 0.0034768, -0.0012488, 0.0010782
7: 0.9790629, 0.9803969, 0.9790217, 0.9804921, -0.0008739, 0.0007545
8: -0.0090121, -0.0075818, -0.0090563, -0.0074797, -0.0009369, 0.0008089
9: 0.0000086, 0.0009534, -0.0000588, 0.0009826, -0.0005343, 0.0006189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005042, upper bound: 0.0004751
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005042, upper bound: 0.0004766
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0005555, -0.0010849, -0.0005853, -0.0002748, 0.0003065
1: -0.0070578, -0.0057203, -0.0070636, -0.0057959, -0.0006974, 0.0007778
2: 0.0306513, 0.0314812, 0.0306477, 0.0314342, -0.0004327, 0.0004825
3: 0.0007792, 0.0023288, 0.0008669, 0.0023355, -0.0009010, 0.0008079
4: -0.0060720, -0.0047115, -0.0060779, -0.0047884, -0.0007094, 0.0007911
5: 0.0114383, 0.0119536, 0.0114360, 0.0119244, -0.0002687, 0.0002997
6: 0.0013771, 0.0033436, 0.0014883, 0.0033521, -0.0011435, 0.0010253
7: 0.9790228, 0.9803990, 0.9791007, 0.9804050, -0.0008002, 0.0007175
8: -0.0090550, -0.0075796, -0.0089716, -0.0075732, -0.0008579, 0.0007692
9: 0.0000072, 0.0009818, 0.0000030, 0.0009266, -0.0005081, 0.0005667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004874
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004875
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0005532, -0.0010882, -0.0005853, -0.0002731, 0.0003126
1: -0.0070819, -0.0057145, -0.0070719, -0.0057958, -0.0006930, 0.0007933
2: 0.0306364, 0.0314847, 0.0306426, 0.0314343, -0.0004299, 0.0004922
3: 0.0007726, 0.0023567, 0.0008667, 0.0023451, -0.0009190, 0.0008028
4: -0.0060965, -0.0047057, -0.0060863, -0.0047883, -0.0007049, 0.0008069
5: 0.0114290, 0.0119558, 0.0114328, 0.0119245, -0.0002670, 0.0003056
6: 0.0013686, 0.0033790, 0.0014881, 0.0033643, -0.0011663, 0.0010188
7: 0.9790170, 0.9804238, 0.9791006, 0.9804134, -0.0008161, 0.0007129
8: -0.0090613, -0.0075530, -0.0089717, -0.0075641, -0.0008750, 0.0007644
9: -0.0000104, 0.0009859, -0.0000031, 0.0009267, -0.0005049, 0.0005780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004969
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004982
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0005555, -0.0010991, -0.0005627, -0.0002762, 0.0003042
1: -0.0070578, -0.0057203, -0.0070996, -0.0057385, -0.0007010, 0.0007719
2: 0.0306513, 0.0314812, 0.0306254, 0.0314698, -0.0004349, 0.0004789
3: 0.0007792, 0.0023288, 0.0008003, 0.0023771, -0.0008942, 0.0008121
4: -0.0060720, -0.0047115, -0.0061145, -0.0047300, -0.0007130, 0.0007852
5: 0.0114383, 0.0119536, 0.0114222, 0.0119466, -0.0002701, 0.0002974
6: 0.0013771, 0.0033436, 0.0014038, 0.0034050, -0.0011349, 0.0010306
7: 0.9790228, 0.9803990, 0.9790417, 0.9804419, -0.0007942, 0.0007212
8: -0.0090550, -0.0075796, -0.0090349, -0.0075336, -0.0008515, 0.0007732
9: 0.0000072, 0.0009818, -0.0000232, 0.0009685, -0.0005108, 0.0005624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0004876
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0004876
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0005532, -0.0011033, -0.0005626, -0.0002770, 0.0003123
1: -0.0070819, -0.0057145, -0.0071103, -0.0057383, -0.0007029, 0.0007926
2: 0.0306364, 0.0314847, 0.0306187, 0.0314700, -0.0004361, 0.0004917
3: 0.0007726, 0.0023567, 0.0008001, 0.0023896, -0.0009182, 0.0008142
4: -0.0060965, -0.0047057, -0.0061254, -0.0047298, -0.0007149, 0.0008062
5: 0.0114290, 0.0119558, 0.0114180, 0.0119467, -0.0002708, 0.0003054
6: 0.0013686, 0.0033790, 0.0014036, 0.0034208, -0.0011653, 0.0010333
7: 0.9790170, 0.9804238, 0.9790414, 0.9804530, -0.0008154, 0.0007231
8: -0.0090613, -0.0075530, -0.0090351, -0.0075217, -0.0008743, 0.0007753
9: -0.0000104, 0.0009859, -0.0000311, 0.0009686, -0.0005121, 0.0005775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0004970
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0004982
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010890, -0.0005531, -0.0010865, -0.0005853, -0.0002797, 0.0003095
1: -0.0070740, -0.0057142, -0.0070678, -0.0057958, -0.0007098, 0.0007853
2: 0.0306413, 0.0314849, 0.0306452, 0.0314343, -0.0004404, 0.0004872
3: 0.0007722, 0.0023474, 0.0008668, 0.0023402, -0.0009097, 0.0008223
4: -0.0060884, -0.0047053, -0.0060821, -0.0047884, -0.0007220, 0.0007988
5: 0.0114320, 0.0119559, 0.0114344, 0.0119245, -0.0002735, 0.0003025
6: 0.0013681, 0.0033673, 0.0014882, 0.0033582, -0.0011545, 0.0010436
7: 0.9790166, 0.9804156, 0.9791006, 0.9804091, -0.0008079, 0.0007303
8: -0.0090617, -0.0075618, -0.0089717, -0.0075687, -0.0008662, 0.0007830
9: -0.0000046, 0.0009862, -0.0000001, 0.0009267, -0.0005172, 0.0005722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004921
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004923
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010977, -0.0005508, -0.0010898, -0.0005853, -0.0002795, 0.0003142
1: -0.0070962, -0.0057082, -0.0070760, -0.0057957, -0.0007093, 0.0007974
2: 0.0306275, 0.0314886, 0.0306400, 0.0314343, -0.0004400, 0.0004947
3: 0.0007653, 0.0023732, 0.0008667, 0.0023498, -0.0009237, 0.0008217
4: -0.0061111, -0.0046993, -0.0060905, -0.0047883, -0.0007215, 0.0008110
5: 0.0114235, 0.0119582, 0.0114313, 0.0119245, -0.0002733, 0.0003072
6: 0.0013594, 0.0034001, 0.0014880, 0.0033703, -0.0011723, 0.0010428
7: 0.9790106, 0.9804385, 0.9791005, 0.9804177, -0.0008203, 0.0007297
8: -0.0090683, -0.0075373, -0.0089718, -0.0075596, -0.0008795, 0.0007824
9: -0.0000208, 0.0009905, -0.0000061, 0.0009268, -0.0005168, 0.0005810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004764, upper bound: 0.0005028
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004764, upper bound: 0.0005029
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010890, -0.0005531, -0.0011014, -0.0005626, -0.0002811, 0.0003045
1: -0.0070740, -0.0057142, -0.0071056, -0.0057384, -0.0007134, 0.0007728
2: 0.0306413, 0.0314849, 0.0306217, 0.0314699, -0.0004426, 0.0004794
3: 0.0007722, 0.0023474, 0.0008002, 0.0023840, -0.0008952, 0.0008264
4: -0.0060884, -0.0047053, -0.0061206, -0.0047299, -0.0007256, 0.0007860
5: 0.0114320, 0.0119559, 0.0114199, 0.0119466, -0.0002749, 0.0002977
6: 0.0013681, 0.0033673, 0.0014037, 0.0034138, -0.0011361, 0.0010489
7: 0.9790166, 0.9804156, 0.9790416, 0.9804481, -0.0007950, 0.0007339
8: -0.0090617, -0.0075618, -0.0090350, -0.0075270, -0.0008524, 0.0007869
9: -0.0000046, 0.0009862, -0.0000276, 0.0009686, -0.0005198, 0.0005630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004755, upper bound: 0.0004922
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004755, upper bound: 0.0004926
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010977, -0.0005508, -0.0011053, -0.0005626, -0.0002820, 0.0003124
1: -0.0070962, -0.0057082, -0.0071154, -0.0057382, -0.0007156, 0.0007929
2: 0.0306275, 0.0314886, 0.0306156, 0.0314700, -0.0004440, 0.0004919
3: 0.0007653, 0.0023732, 0.0008000, 0.0023955, -0.0009185, 0.0008290
4: -0.0061111, -0.0046993, -0.0061306, -0.0047297, -0.0007279, 0.0008065
5: 0.0114235, 0.0119582, 0.0114161, 0.0119467, -0.0002757, 0.0003055
6: 0.0013594, 0.0034001, 0.0014034, 0.0034283, -0.0011657, 0.0010521
7: 0.9790106, 0.9804385, 0.9790414, 0.9804582, -0.0008157, 0.0007362
8: -0.0090683, -0.0075373, -0.0090352, -0.0075161, -0.0008745, 0.0007893
9: -0.0000208, 0.0009905, -0.0000348, 0.0009687, -0.0005214, 0.0005777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004772, upper bound: 0.0005028
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004772, upper bound: 0.0005029
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010828, -0.0005531, -0.0010996, -0.0005764, -0.0002835, 0.0003310
1: -0.0070584, -0.0057142, -0.0071010, -0.0057733, -0.0007194, 0.0008399
2: 0.0306510, 0.0314849, 0.0306246, 0.0314482, -0.0004463, 0.0005211
3: 0.0007722, 0.0023294, 0.0008407, 0.0023787, -0.0009730, 0.0008334
4: -0.0060726, -0.0047053, -0.0061159, -0.0047655, -0.0007318, 0.0008543
5: 0.0114380, 0.0119559, 0.0114216, 0.0119331, -0.0002772, 0.0003236
6: 0.0013681, 0.0033444, 0.0014551, 0.0034070, -0.0012348, 0.0010577
7: 0.9790166, 0.9803995, 0.9790776, 0.9804433, -0.0008641, 0.0007402
8: -0.0090618, -0.0075790, -0.0089965, -0.0075321, -0.0009264, 0.0007936
9: 0.0000068, 0.0009862, -0.0000242, 0.0009431, -0.0005242, 0.0006120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004998
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004998
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010923, -0.0005515, -0.0011028, -0.0005763, -0.0002805, 0.0003387
1: -0.0070825, -0.0057101, -0.0071090, -0.0057731, -0.0007117, 0.0008596
2: 0.0306360, 0.0314875, 0.0306196, 0.0314484, -0.0004415, 0.0005333
3: 0.0007674, 0.0023573, 0.0008405, 0.0023880, -0.0009958, 0.0008245
4: -0.0060971, -0.0047011, -0.0061241, -0.0047653, -0.0007239, 0.0008743
5: 0.0114288, 0.0119575, 0.0114186, 0.0119332, -0.0002742, 0.0003312
6: 0.0013621, 0.0033798, 0.0014548, 0.0034188, -0.0012637, 0.0010464
7: 0.9790124, 0.9804243, 0.9790773, 0.9804516, -0.0008843, 0.0007322
8: -0.0090663, -0.0075525, -0.0089967, -0.0075232, -0.0009481, 0.0007850
9: -0.0000108, 0.0009892, -0.0000301, 0.0009432, -0.0005186, 0.0006263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004721, upper bound: 0.0005106
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004721, upper bound: 0.0005120
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010828, -0.0005531, -0.0011131, -0.0005552, -0.0002845, 0.0003280
1: -0.0070584, -0.0057142, -0.0071353, -0.0057194, -0.0007220, 0.0008325
2: 0.0306510, 0.0314849, 0.0306033, 0.0314817, -0.0004479, 0.0005165
3: 0.0007722, 0.0023294, 0.0007783, 0.0024184, -0.0009644, 0.0008364
4: -0.0060726, -0.0047053, -0.0061508, -0.0047107, -0.0007344, 0.0008468
5: 0.0114380, 0.0119559, 0.0114084, 0.0119539, -0.0002782, 0.0003207
6: 0.0013681, 0.0033444, 0.0013759, 0.0034574, -0.0012239, 0.0010614
7: 0.9790166, 0.9803995, 0.9790220, 0.9804786, -0.0008564, 0.0007428
8: -0.0090618, -0.0075790, -0.0090559, -0.0074942, -0.0009182, 0.0007963
9: 0.0000068, 0.0009862, -0.0000492, 0.0009823, -0.0005260, 0.0006065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0004999
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0004999
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010923, -0.0005515, -0.0011174, -0.0005551, -0.0002834, 0.0003373
1: -0.0070825, -0.0057101, -0.0071460, -0.0057192, -0.0007192, 0.0008559
2: 0.0306360, 0.0314875, 0.0305966, 0.0314818, -0.0004462, 0.0005310
3: 0.0007674, 0.0023573, 0.0007780, 0.0024309, -0.0009915, 0.0008331
4: -0.0060971, -0.0047011, -0.0061618, -0.0047104, -0.0007315, 0.0008706
5: 0.0114288, 0.0119575, 0.0114043, 0.0119540, -0.0002771, 0.0003298
6: 0.0013621, 0.0033798, 0.0013755, 0.0034733, -0.0012583, 0.0010573
7: 0.9790124, 0.9804243, 0.9790218, 0.9804897, -0.0008805, 0.0007399
8: -0.0090663, -0.0075525, -0.0090562, -0.0074823, -0.0009441, 0.0007932
9: -0.0000108, 0.0009892, -0.0000571, 0.0009825, -0.0005240, 0.0006236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0005108
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0005121
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010892, -0.0005510, -0.0011007, -0.0005764, -0.0002882, 0.0003325
1: -0.0070746, -0.0057089, -0.0071037, -0.0057733, -0.0007314, 0.0008437
2: 0.0306409, 0.0314882, 0.0306229, 0.0314483, -0.0004538, 0.0005234
3: 0.0007661, 0.0023481, 0.0008406, 0.0023819, -0.0009774, 0.0008473
4: -0.0060890, -0.0046999, -0.0061187, -0.0047654, -0.0007440, 0.0008582
5: 0.0114318, 0.0119580, 0.0114206, 0.0119332, -0.0002818, 0.0003251
6: 0.0013604, 0.0033682, 0.0014550, 0.0034110, -0.0012405, 0.0010753
7: 0.9790112, 0.9804161, 0.9790775, 0.9804462, -0.0008680, 0.0007525
8: -0.0090676, -0.0075612, -0.0089965, -0.0075290, -0.0009307, 0.0008068
9: -0.0000050, 0.0009900, -0.0000262, 0.0009431, -0.0005329, 0.0006147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0005041
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0005042
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0005493, -0.0011038, -0.0005763, -0.0002870, 0.0003395
1: -0.0070969, -0.0057045, -0.0071115, -0.0057731, -0.0007284, 0.0008616
2: 0.0306271, 0.0314909, 0.0306180, 0.0314484, -0.0004519, 0.0005345
3: 0.0007610, 0.0023740, 0.0008404, 0.0023910, -0.0009981, 0.0008438
4: -0.0061117, -0.0046954, -0.0061267, -0.0047652, -0.0007409, 0.0008764
5: 0.0114232, 0.0119597, 0.0114176, 0.0119332, -0.0002806, 0.0003319
6: 0.0013539, 0.0034010, 0.0014547, 0.0034226, -0.0012667, 0.0010708
7: 0.9790066, 0.9804391, 0.9790772, 0.9804542, -0.0008864, 0.0007493
8: -0.0090724, -0.0075366, -0.0089968, -0.0075204, -0.0009504, 0.0008034
9: -0.0000213, 0.0009932, -0.0000319, 0.0009433, -0.0005307, 0.0006278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0005162
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0005176
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010892, -0.0005510, -0.0011140, -0.0005552, -0.0002886, 0.0003268
1: -0.0070746, -0.0057089, -0.0071376, -0.0057194, -0.0007325, 0.0008294
2: 0.0306409, 0.0314882, 0.0306018, 0.0314817, -0.0004544, 0.0005146
3: 0.0007661, 0.0023481, 0.0007782, 0.0024212, -0.0009608, 0.0008486
4: -0.0060890, -0.0046999, -0.0061532, -0.0047106, -0.0007451, 0.0008437
5: 0.0114318, 0.0119580, 0.0114075, 0.0119539, -0.0002822, 0.0003196
6: 0.0013604, 0.0033682, 0.0013757, 0.0034609, -0.0012194, 0.0010769
7: 0.9790112, 0.9804161, 0.9790219, 0.9804810, -0.0008533, 0.0007536
8: -0.0090676, -0.0075612, -0.0090560, -0.0074916, -0.0009149, 0.0008080
9: -0.0000050, 0.0009900, -0.0000509, 0.0009824, -0.0005337, 0.0006043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0005042
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004755, upper bound: 0.0005043
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0005493, -0.0011183, -0.0005551, -0.0002890, 0.0003368
1: -0.0070969, -0.0057045, -0.0071484, -0.0057191, -0.0007333, 0.0008548
2: 0.0306271, 0.0314909, 0.0305951, 0.0314819, -0.0004550, 0.0005303
3: 0.0007610, 0.0023740, 0.0007779, 0.0024337, -0.0009902, 0.0008495
4: -0.0061117, -0.0046954, -0.0061641, -0.0047103, -0.0007459, 0.0008695
5: 0.0114232, 0.0119597, 0.0114034, 0.0119540, -0.0002825, 0.0003293
6: 0.0013539, 0.0034010, 0.0013754, 0.0034768, -0.0012567, 0.0010782
7: 0.9790066, 0.9804391, 0.9790217, 0.9804921, -0.0008794, 0.0007545
8: -0.0090724, -0.0075366, -0.0090563, -0.0074797, -0.0009429, 0.0008089
9: -0.0000213, 0.0009932, -0.0000588, 0.0009826, -0.0005343, 0.0006228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0005165
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0005177
time: 0.56 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.26 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004727
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004727
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0004794
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0004804
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005122, upper bound: 0.0004637
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005122, upper bound: 0.0004637
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005041, upper bound: 0.0004655
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005041, upper bound: 0.0004670
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004757
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004758
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004942, upper bound: 0.0004842
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004942, upper bound: 0.0004843
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005106, upper bound: 0.0004628
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005106, upper bound: 0.0004630
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005120, upper bound: 0.0004662
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005120, upper bound: 0.0004667
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004861
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004861
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004885, upper bound: 0.0004932
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004885, upper bound: 0.0004945
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005122, upper bound: 0.0004721
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005122, upper bound: 0.0004721
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005040, upper bound: 0.0004755
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005040, upper bound: 0.0004770
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004886
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004930, upper bound: 0.0004886
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0004957
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004886, upper bound: 0.0004969
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005106, upper bound: 0.0004722
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005106, upper bound: 0.0004722
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005042, upper bound: 0.0004751
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0005042, upper bound: 0.0004766
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004874
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004875
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004969
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0004982
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0004876
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0004876
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0004970
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0004982
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004921
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004923
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004764, upper bound: 0.0005028
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004764, upper bound: 0.0005029
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004755, upper bound: 0.0004922
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004755, upper bound: 0.0004926
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004772, upper bound: 0.0005028
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004772, upper bound: 0.0005029
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004998
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0004998
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004721, upper bound: 0.0005106
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004721, upper bound: 0.0005120
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0004999
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0004999
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0005108
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0005121
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0005041
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004749, upper bound: 0.0005042
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0005162
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004722, upper bound: 0.0005176
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004754, upper bound: 0.0005042
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004755, upper bound: 0.0005043
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0005165
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 7, lower bound: -0.0004731, upper bound: 0.0005177

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010704, -0.0005751, -0.0010769, -0.0005854, -0.0002621, 0.0002825
1: -0.0070269, -0.0057699, -0.0070434, -0.0057960, -0.0006651, 0.0007169
2: 0.0306705, 0.0314503, 0.0306603, 0.0314341, -0.0004126, 0.0004448
3: 0.0008368, 0.0022930, 0.0008670, 0.0023120, -0.0008305, 0.0007704
4: -0.0060406, -0.0047620, -0.0060573, -0.0047886, -0.0006765, 0.0007292
5: 0.0114502, 0.0119345, 0.0114438, 0.0119244, -0.0002562, 0.0002762
6: 0.0014501, 0.0032982, 0.0014885, 0.0033223, -0.0010540, 0.0009778
7: 0.9790741, 0.9803672, 0.9791008, 0.9803841, -0.0007376, 0.0006842
8: -0.0090002, -0.0076137, -0.0089714, -0.0075956, -0.0007908, 0.0007336
9: 0.0000297, 0.0009456, 0.0000177, 0.0009265, -0.0004846, 0.0005224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004727
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004727
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010704, -0.0005751, -0.0010809, -0.0005854, -0.0002599, 0.0002844
1: -0.0070269, -0.0057699, -0.0070535, -0.0057961, -0.0006595, 0.0007217
2: 0.0306705, 0.0314503, 0.0306540, 0.0314341, -0.0004091, 0.0004477
3: 0.0008368, 0.0022930, 0.0008671, 0.0023238, -0.0008360, 0.0007640
4: -0.0060406, -0.0047620, -0.0060677, -0.0047886, -0.0006708, 0.0007341
5: 0.0114502, 0.0119345, 0.0114399, 0.0119244, -0.0002541, 0.0002780
6: 0.0014501, 0.0032982, 0.0014886, 0.0033373, -0.0010610, 0.0009696
7: 0.9790741, 0.9803672, 0.9791009, 0.9803945, -0.0007424, 0.0006785
8: -0.0090002, -0.0076137, -0.0089714, -0.0075844, -0.0007960, 0.0007274
9: 0.0000297, 0.0009456, 0.0000103, 0.0009265, -0.0004805, 0.0005258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004727
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004727
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010772, -0.0005729, -0.0010767, -0.0005874, -0.0002694, 0.0002813
1: -0.0070441, -0.0057643, -0.0070428, -0.0058013, -0.0006837, 0.0007138
2: 0.0306598, 0.0314538, 0.0306607, 0.0314309, -0.0004242, 0.0004429
3: 0.0008303, 0.0023129, 0.0008731, 0.0023113, -0.0008269, 0.0007920
4: -0.0060581, -0.0047563, -0.0060567, -0.0047939, -0.0006954, 0.0007261
5: 0.0114435, 0.0119366, 0.0114441, 0.0119224, -0.0002634, 0.0002750
6: 0.0014419, 0.0033235, 0.0014962, 0.0033215, -0.0010495, 0.0010052
7: 0.9790682, 0.9803848, 0.9791062, 0.9803835, -0.0007344, 0.0007034
8: -0.0090064, -0.0075947, -0.0089657, -0.0075962, -0.0007874, 0.0007541
9: 0.0000172, 0.0009496, 0.0000181, 0.0009227, -0.0004981, 0.0005201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004823, upper bound: 0.0004794
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004823, upper bound: 0.0004794
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010772, -0.0005729, -0.0010836, -0.0005857, -0.0002581, 0.0002804
1: -0.0070441, -0.0057643, -0.0070604, -0.0057968, -0.0006550, 0.0007114
2: 0.0306598, 0.0314538, 0.0306497, 0.0314337, -0.0004064, 0.0004414
3: 0.0008303, 0.0023129, 0.0008679, 0.0023317, -0.0008242, 0.0007588
4: -0.0060581, -0.0047563, -0.0060747, -0.0047893, -0.0006663, 0.0007237
5: 0.0114435, 0.0119366, 0.0114373, 0.0119241, -0.0002524, 0.0002741
6: 0.0014419, 0.0033235, 0.0014896, 0.0033474, -0.0010460, 0.0009630
7: 0.9790682, 0.9803848, 0.9791017, 0.9804016, -0.0007319, 0.0006739
8: -0.0090064, -0.0075947, -0.0089706, -0.0075768, -0.0007847, 0.0007225
9: 0.0000172, 0.0009496, 0.0000053, 0.0009260, -0.0004773, 0.0005184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004823, upper bound: 0.0004804
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004823, upper bound: 0.0004804
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010704, -0.0005751, -0.0010908, -0.0005657, -0.0002794, 0.0002968
1: -0.0070269, -0.0057699, -0.0070786, -0.0057461, -0.0007090, 0.0007532
2: 0.0306705, 0.0314503, 0.0306384, 0.0314651, -0.0004399, 0.0004673
3: 0.0008368, 0.0022930, 0.0008092, 0.0023528, -0.0008725, 0.0008214
4: -0.0060406, -0.0047620, -0.0060932, -0.0047378, -0.0007212, 0.0007661
5: 0.0114502, 0.0119345, 0.0114303, 0.0119436, -0.0002732, 0.0002902
6: 0.0014501, 0.0032982, 0.0014151, 0.0033742, -0.0011073, 0.0010424
7: 0.9790741, 0.9803672, 0.9790494, 0.9804203, -0.0007749, 0.0007294
8: -0.0090002, -0.0076137, -0.0090265, -0.0075567, -0.0008308, 0.0007821
9: 0.0000297, 0.0009456, -0.0000080, 0.0009629, -0.0005166, 0.0005488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004637
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004637
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010704, -0.0005751, -0.0010967, -0.0005627, -0.0002794, 0.0003009
1: -0.0070269, -0.0057699, -0.0070936, -0.0057386, -0.0007090, 0.0007636
2: 0.0306705, 0.0314503, 0.0306291, 0.0314698, -0.0004399, 0.0004738
3: 0.0008368, 0.0022930, 0.0008004, 0.0023702, -0.0008846, 0.0008213
4: -0.0060406, -0.0047620, -0.0061084, -0.0047301, -0.0007211, 0.0007767
5: 0.0114502, 0.0119345, 0.0114245, 0.0119465, -0.0002732, 0.0002942
6: 0.0014501, 0.0032982, 0.0014040, 0.0033962, -0.0011227, 0.0010424
7: 0.9790741, 0.9803672, 0.9790418, 0.9804357, -0.0007856, 0.0007294
8: -0.0090002, -0.0076137, -0.0090348, -0.0075402, -0.0008423, 0.0007820
9: 0.0000297, 0.0009456, -0.0000189, 0.0009684, -0.0005166, 0.0005564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004637
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004637
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010772, -0.0005729, -0.0010897, -0.0005659, -0.0002934, 0.0002969
1: -0.0070441, -0.0057643, -0.0070759, -0.0057465, -0.0007445, 0.0007534
2: 0.0306598, 0.0314538, 0.0306401, 0.0314649, -0.0004619, 0.0004674
3: 0.0008303, 0.0023129, 0.0008096, 0.0023496, -0.0008728, 0.0008625
4: -0.0060581, -0.0047563, -0.0060904, -0.0047382, -0.0007573, 0.0007664
5: 0.0114435, 0.0119366, 0.0114313, 0.0119435, -0.0002868, 0.0002903
6: 0.0014419, 0.0033235, 0.0014156, 0.0033701, -0.0011077, 0.0010946
7: 0.9790682, 0.9803848, 0.9790499, 0.9804175, -0.0007751, 0.0007660
8: -0.0090064, -0.0075947, -0.0090261, -0.0075598, -0.0008310, 0.0008212
9: 0.0000172, 0.0009496, -0.0000059, 0.0009626, -0.0005425, 0.0005490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004974, upper bound: 0.0004655
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004974, upper bound: 0.0004655
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010772, -0.0005729, -0.0010991, -0.0005630, -0.0002814, 0.0002961
1: -0.0070441, -0.0057643, -0.0070998, -0.0057392, -0.0007140, 0.0007515
2: 0.0306598, 0.0314538, 0.0306253, 0.0314694, -0.0004430, 0.0004662
3: 0.0008303, 0.0023129, 0.0008011, 0.0023774, -0.0008706, 0.0008271
4: -0.0060581, -0.0047563, -0.0061147, -0.0047307, -0.0007262, 0.0007644
5: 0.0114435, 0.0119366, 0.0114221, 0.0119463, -0.0002751, 0.0002895
6: 0.0014419, 0.0033235, 0.0014049, 0.0034053, -0.0011049, 0.0010497
7: 0.9790682, 0.9803848, 0.9790424, 0.9804422, -0.0007732, 0.0007345
8: -0.0090064, -0.0075947, -0.0090341, -0.0075333, -0.0008289, 0.0007875
9: 0.0000172, 0.0009496, -0.0000234, 0.0009680, -0.0005202, 0.0005476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004974, upper bound: 0.0004670
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004974, upper bound: 0.0004670
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010741, -0.0005743, -0.0010769, -0.0005854, -0.0002646, 0.0002788
1: -0.0070362, -0.0057679, -0.0070434, -0.0057960, -0.0006714, 0.0007075
2: 0.0306647, 0.0314516, 0.0306603, 0.0314341, -0.0004165, 0.0004390
3: 0.0008344, 0.0023037, 0.0008670, 0.0023120, -0.0008197, 0.0007778
4: -0.0060500, -0.0047599, -0.0060573, -0.0047886, -0.0006829, 0.0007197
5: 0.0114466, 0.0119352, 0.0114438, 0.0119244, -0.0002587, 0.0002726
6: 0.0014471, 0.0033118, 0.0014885, 0.0033223, -0.0010402, 0.0009871
7: 0.9790719, 0.9803767, 0.9791008, 0.9803841, -0.0007279, 0.0006907
8: -0.0090025, -0.0076035, -0.0089714, -0.0075956, -0.0007804, 0.0007406
9: 0.0000229, 0.0009471, 0.0000177, 0.0009265, -0.0004892, 0.0005155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004757
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004757
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010741, -0.0005743, -0.0010809, -0.0005854, -0.0002648, 0.0002855
1: -0.0070362, -0.0057679, -0.0070535, -0.0057961, -0.0006720, 0.0007245
2: 0.0306647, 0.0314516, 0.0306540, 0.0314341, -0.0004169, 0.0004495
3: 0.0008344, 0.0023037, 0.0008671, 0.0023238, -0.0008393, 0.0007785
4: -0.0060500, -0.0047599, -0.0060677, -0.0047886, -0.0006835, 0.0007370
5: 0.0114466, 0.0119352, 0.0114399, 0.0119244, -0.0002589, 0.0002791
6: 0.0014471, 0.0033118, 0.0014886, 0.0033373, -0.0010652, 0.0009880
7: 0.9790719, 0.9803767, 0.9791009, 0.9803945, -0.0007454, 0.0006914
8: -0.0090025, -0.0076035, -0.0089714, -0.0075844, -0.0007992, 0.0007412
9: 0.0000229, 0.0009471, 0.0000103, 0.0009265, -0.0004896, 0.0005279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004758
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004758
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010815, -0.0005730, -0.0010800, -0.0005853, -0.0002645, 0.0002880
1: -0.0070550, -0.0057646, -0.0070512, -0.0057959, -0.0006713, 0.0007308
2: 0.0306531, 0.0314536, 0.0306554, 0.0314342, -0.0004165, 0.0004534
3: 0.0008306, 0.0023254, 0.0008668, 0.0023210, -0.0008466, 0.0007776
4: -0.0060691, -0.0047566, -0.0060653, -0.0047884, -0.0006828, 0.0007433
5: 0.0114394, 0.0119365, 0.0114408, 0.0119245, -0.0002586, 0.0002816
6: 0.0014423, 0.0033394, 0.0014882, 0.0033338, -0.0010744, 0.0009869
7: 0.9790685, 0.9803960, 0.9791006, 0.9803922, -0.0007518, 0.0006906
8: -0.0090061, -0.0075828, -0.0089716, -0.0075870, -0.0008061, 0.0007404
9: 0.0000093, 0.0009494, 0.0000120, 0.0009267, -0.0004891, 0.0005325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004843
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004843
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010815, -0.0005730, -0.0010842, -0.0005854, -0.0002647, 0.0002930
1: -0.0070550, -0.0057646, -0.0070618, -0.0057960, -0.0006716, 0.0007436
2: 0.0306531, 0.0314536, 0.0306489, 0.0314341, -0.0004167, 0.0004613
3: 0.0008306, 0.0023254, 0.0008670, 0.0023333, -0.0008614, 0.0007780
4: -0.0060691, -0.0047566, -0.0060760, -0.0047885, -0.0006831, 0.0007564
5: 0.0114394, 0.0119365, 0.0114367, 0.0119244, -0.0002587, 0.0002865
6: 0.0014423, 0.0033394, 0.0014884, 0.0033494, -0.0010933, 0.0009874
7: 0.9790685, 0.9803960, 0.9791008, 0.9804030, -0.0007650, 0.0006909
8: -0.0090061, -0.0075828, -0.0089715, -0.0075753, -0.0008202, 0.0007408
9: 0.0000093, 0.0009494, 0.0000043, 0.0009266, -0.0004893, 0.0005418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004843
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004843
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010741, -0.0005743, -0.0010908, -0.0005657, -0.0002819, 0.0002931
1: -0.0070362, -0.0057679, -0.0070786, -0.0057461, -0.0007154, 0.0007438
2: 0.0306647, 0.0314516, 0.0306384, 0.0314651, -0.0004438, 0.0004614
3: 0.0008344, 0.0023037, 0.0008092, 0.0023528, -0.0008616, 0.0008287
4: -0.0060500, -0.0047599, -0.0060932, -0.0047378, -0.0007276, 0.0007566
5: 0.0114466, 0.0119352, 0.0114303, 0.0119436, -0.0002756, 0.0002866
6: 0.0014471, 0.0033118, 0.0014151, 0.0033742, -0.0010935, 0.0010517
7: 0.9790719, 0.9803767, 0.9790494, 0.9804203, -0.0007652, 0.0007360
8: -0.0090025, -0.0076035, -0.0090265, -0.0075567, -0.0008204, 0.0007891
9: 0.0000229, 0.0009471, -0.0000080, 0.0009629, -0.0005212, 0.0005419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0004628
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0004628
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010741, -0.0005743, -0.0010967, -0.0005627, -0.0002823, 0.0002998
1: -0.0070362, -0.0057679, -0.0070936, -0.0057386, -0.0007163, 0.0007608
2: 0.0306647, 0.0314516, 0.0306291, 0.0314698, -0.0004444, 0.0004720
3: 0.0008344, 0.0023037, 0.0008004, 0.0023702, -0.0008814, 0.0008298
4: -0.0060500, -0.0047599, -0.0061084, -0.0047301, -0.0007286, 0.0007739
5: 0.0114466, 0.0119352, 0.0114245, 0.0119465, -0.0002760, 0.0002931
6: 0.0014471, 0.0033118, 0.0014040, 0.0033962, -0.0011186, 0.0010532
7: 0.9790719, 0.9803767, 0.9790418, 0.9804357, -0.0007827, 0.0007370
8: -0.0090025, -0.0076035, -0.0090348, -0.0075402, -0.0008392, 0.0007901
9: 0.0000229, 0.0009471, -0.0000189, 0.0009684, -0.0005219, 0.0005544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0004630
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0004630
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010815, -0.0005730, -0.0010950, -0.0005656, -0.0002824, 0.0003015
1: -0.0070550, -0.0057646, -0.0070892, -0.0057459, -0.0007167, 0.0007652
2: 0.0306531, 0.0314536, 0.0306319, 0.0314653, -0.0004446, 0.0004747
3: 0.0008306, 0.0023254, 0.0008089, 0.0023651, -0.0008864, 0.0008303
4: -0.0060691, -0.0047566, -0.0061039, -0.0047375, -0.0007290, 0.0007783
5: 0.0114394, 0.0119365, 0.0114262, 0.0119437, -0.0002761, 0.0002948
6: 0.0014423, 0.0033394, 0.0014147, 0.0033897, -0.0011250, 0.0010537
7: 0.9790685, 0.9803960, 0.9790492, 0.9804312, -0.0007872, 0.0007373
8: -0.0090061, -0.0075828, -0.0090268, -0.0075451, -0.0008440, 0.0007905
9: 0.0000093, 0.0009494, -0.0000157, 0.0009631, -0.0005222, 0.0005575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004662
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004662
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010815, -0.0005730, -0.0011005, -0.0005627, -0.0002827, 0.0003066
1: -0.0070550, -0.0057646, -0.0071034, -0.0057384, -0.0007174, 0.0007781
2: 0.0306531, 0.0314536, 0.0306231, 0.0314699, -0.0004451, 0.0004828
3: 0.0008306, 0.0023254, 0.0008002, 0.0023815, -0.0009014, 0.0008311
4: -0.0060691, -0.0047566, -0.0061183, -0.0047299, -0.0007297, 0.0007915
5: 0.0114394, 0.0119365, 0.0114207, 0.0119466, -0.0002764, 0.0002998
6: 0.0014423, 0.0033394, 0.0014037, 0.0034105, -0.0011440, 0.0010547
7: 0.9790685, 0.9803960, 0.9790416, 0.9804457, -0.0008006, 0.0007381
8: -0.0090061, -0.0075828, -0.0090350, -0.0075294, -0.0008583, 0.0007913
9: 0.0000093, 0.0009494, -0.0000260, 0.0009685, -0.0005227, 0.0005670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004667
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004667
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010708, -0.0005722, -0.0010916, -0.0005768, -0.0002707, 0.0003080
1: -0.0070279, -0.0057626, -0.0070807, -0.0057744, -0.0006870, 0.0007816
2: 0.0306699, 0.0314549, 0.0306371, 0.0314476, -0.0004262, 0.0004849
3: 0.0008283, 0.0022941, 0.0008419, 0.0023552, -0.0009054, 0.0007959
4: -0.0060416, -0.0047546, -0.0060953, -0.0047666, -0.0006988, 0.0007950
5: 0.0114498, 0.0119373, 0.0114295, 0.0119327, -0.0002647, 0.0003011
6: 0.0014394, 0.0032996, 0.0014567, 0.0033772, -0.0011491, 0.0010101
7: 0.9790665, 0.9803681, 0.9790786, 0.9804224, -0.0008041, 0.0007068
8: -0.0090083, -0.0076126, -0.0089953, -0.0075545, -0.0008621, 0.0007578
9: 0.0000290, 0.0009509, -0.0000095, 0.0009423, -0.0005006, 0.0005695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004861
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004727
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010708, -0.0005722, -0.0010949, -0.0005765, -0.0002685, 0.0003088
1: -0.0070279, -0.0057626, -0.0070890, -0.0057736, -0.0006814, 0.0007837
2: 0.0306699, 0.0314549, 0.0306320, 0.0314480, -0.0004228, 0.0004862
3: 0.0008283, 0.0022941, 0.0008411, 0.0023648, -0.0009079, 0.0007894
4: -0.0060416, -0.0047546, -0.0061037, -0.0047658, -0.0006931, 0.0007972
5: 0.0114498, 0.0119373, 0.0114263, 0.0119330, -0.0002625, 0.0003019
6: 0.0014394, 0.0032996, 0.0014556, 0.0033894, -0.0011522, 0.0010019
7: 0.9790665, 0.9803681, 0.9790778, 0.9804310, -0.0008063, 0.0007011
8: -0.0090083, -0.0076126, -0.0089961, -0.0075453, -0.0008644, 0.0007517
9: 0.0000290, 0.0009509, -0.0000155, 0.0009429, -0.0004965, 0.0005710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004861
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004727
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010776, -0.0005710, -0.0010913, -0.0005780, -0.0002760, 0.0003089
1: -0.0070450, -0.0057597, -0.0070798, -0.0057774, -0.0007005, 0.0007839
2: 0.0306592, 0.0314567, 0.0306377, 0.0314457, -0.0004346, 0.0004863
3: 0.0008249, 0.0023139, 0.0008454, 0.0023542, -0.0009081, 0.0008114
4: -0.0060590, -0.0047516, -0.0060944, -0.0047696, -0.0007125, 0.0007973
5: 0.0114432, 0.0119384, 0.0114298, 0.0119316, -0.0002699, 0.0003020
6: 0.0014350, 0.0033248, 0.0014610, 0.0033760, -0.0011525, 0.0010298
7: 0.9790635, 0.9803858, 0.9790816, 0.9804216, -0.0008065, 0.0007206
8: -0.0090115, -0.0075937, -0.0089920, -0.0075554, -0.0008646, 0.0007726
9: 0.0000165, 0.0009530, -0.0000089, 0.0009401, -0.0005104, 0.0005711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004757, upper bound: 0.0004932
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004757, upper bound: 0.0004794
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010776, -0.0005710, -0.0010981, -0.0005769, -0.0002653, 0.0003053
1: -0.0070450, -0.0057597, -0.0070972, -0.0057745, -0.0006732, 0.0007748
2: 0.0306592, 0.0314567, 0.0306269, 0.0314475, -0.0004177, 0.0004807
3: 0.0008249, 0.0023139, 0.0008421, 0.0023744, -0.0008976, 0.0007799
4: -0.0060590, -0.0047516, -0.0061121, -0.0047667, -0.0006848, 0.0007881
5: 0.0114432, 0.0119384, 0.0114231, 0.0119327, -0.0002594, 0.0002985
6: 0.0014350, 0.0033248, 0.0014568, 0.0034015, -0.0011391, 0.0009898
7: 0.9790635, 0.9803858, 0.9790788, 0.9804395, -0.0007971, 0.0006926
8: -0.0090115, -0.0075937, -0.0089952, -0.0075362, -0.0008546, 0.0007426
9: 0.0000165, 0.0009530, -0.0000215, 0.0009422, -0.0004905, 0.0005645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004757, upper bound: 0.0004945
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004757, upper bound: 0.0004804
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010708, -0.0005722, -0.0011049, -0.0005571, -0.0002862, 0.0003168
1: -0.0070279, -0.0057626, -0.0071144, -0.0057244, -0.0007263, 0.0008040
2: 0.0306699, 0.0314549, 0.0306162, 0.0314786, -0.0004506, 0.0004988
3: 0.0008283, 0.0022941, 0.0007840, 0.0023943, -0.0009314, 0.0008414
4: -0.0060416, -0.0047546, -0.0061296, -0.0047157, -0.0007388, 0.0008178
5: 0.0114498, 0.0119373, 0.0114165, 0.0119520, -0.0002798, 0.0003098
6: 0.0014394, 0.0032996, 0.0013831, 0.0034268, -0.0011820, 0.0010679
7: 0.9790665, 0.9803681, 0.9790272, 0.9804571, -0.0008271, 0.0007472
8: -0.0090083, -0.0076126, -0.0090505, -0.0075172, -0.0008868, 0.0008012
9: 0.0000290, 0.0009509, -0.0000341, 0.0009788, -0.0005292, 0.0005858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004721
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004637
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010708, -0.0005722, -0.0011088, -0.0005553, -0.0002871, 0.0003219
1: -0.0070279, -0.0057626, -0.0071243, -0.0057197, -0.0007286, 0.0008168
2: 0.0306699, 0.0314549, 0.0306101, 0.0314815, -0.0004521, 0.0005068
3: 0.0008283, 0.0022941, 0.0007786, 0.0024057, -0.0009462, 0.0008441
4: -0.0060416, -0.0047546, -0.0061396, -0.0047109, -0.0007411, 0.0008308
5: 0.0114498, 0.0119373, 0.0114127, 0.0119538, -0.0002807, 0.0003147
6: 0.0014394, 0.0032996, 0.0013762, 0.0034413, -0.0012009, 0.0010713
7: 0.9790665, 0.9803681, 0.9790223, 0.9804674, -0.0008403, 0.0007496
8: -0.0090083, -0.0076126, -0.0090557, -0.0075063, -0.0009010, 0.0008037
9: 0.0000290, 0.0009509, -0.0000412, 0.0009822, -0.0005309, 0.0005951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004721
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004637
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010776, -0.0005710, -0.0011033, -0.0005572, -0.0002997, 0.0003194
1: -0.0070450, -0.0057597, -0.0071104, -0.0057244, -0.0007605, 0.0008105
2: 0.0306592, 0.0314567, 0.0306187, 0.0314786, -0.0004718, 0.0005028
3: 0.0008249, 0.0023139, 0.0007841, 0.0023896, -0.0009389, 0.0008810
4: -0.0060590, -0.0047516, -0.0061255, -0.0047157, -0.0007736, 0.0008244
5: 0.0114432, 0.0119384, 0.0114180, 0.0119520, -0.0002930, 0.0003123
6: 0.0014350, 0.0033248, 0.0013832, 0.0034209, -0.0011916, 0.0011181
7: 0.9790635, 0.9803858, 0.9790272, 0.9804531, -0.0008338, 0.0007824
8: -0.0090115, -0.0075937, -0.0090504, -0.0075217, -0.0008940, 0.0008389
9: 0.0000165, 0.0009530, -0.0000311, 0.0009787, -0.0005541, 0.0005905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004921, upper bound: 0.0004755
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004921, upper bound: 0.0004655
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010776, -0.0005710, -0.0011131, -0.0005556, -0.0002880, 0.0003163
1: -0.0070450, -0.0057597, -0.0071353, -0.0057205, -0.0007307, 0.0008028
2: 0.0306592, 0.0314567, 0.0306033, 0.0314810, -0.0004534, 0.0004980
3: 0.0008249, 0.0023139, 0.0007795, 0.0024185, -0.0009300, 0.0008465
4: -0.0060590, -0.0047516, -0.0061508, -0.0047117, -0.0007433, 0.0008165
5: 0.0114432, 0.0119384, 0.0114084, 0.0119535, -0.0002815, 0.0003093
6: 0.0014350, 0.0033248, 0.0013774, 0.0034575, -0.0011802, 0.0010744
7: 0.9790635, 0.9803858, 0.9790231, 0.9804786, -0.0008259, 0.0007518
8: -0.0090115, -0.0075937, -0.0090548, -0.0074942, -0.0008855, 0.0008060
9: 0.0000165, 0.0009530, -0.0000492, 0.0009816, -0.0005324, 0.0005849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004921, upper bound: 0.0004770
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004921, upper bound: 0.0004670
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010744, -0.0005718, -0.0010916, -0.0005768, -0.0002733, 0.0003044
1: -0.0070371, -0.0057616, -0.0070807, -0.0057744, -0.0006936, 0.0007725
2: 0.0306642, 0.0314555, 0.0306371, 0.0314476, -0.0004303, 0.0004792
3: 0.0008271, 0.0023047, 0.0008419, 0.0023552, -0.0008949, 0.0008035
4: -0.0060509, -0.0047536, -0.0060953, -0.0047666, -0.0007055, 0.0007857
5: 0.0114463, 0.0119377, 0.0114295, 0.0119327, -0.0002672, 0.0002976
6: 0.0014379, 0.0033131, 0.0014567, 0.0033772, -0.0011357, 0.0010198
7: 0.9790654, 0.9803777, 0.9790786, 0.9804224, -0.0007947, 0.0007136
8: -0.0090094, -0.0076025, -0.0089953, -0.0075545, -0.0008520, 0.0007651
9: 0.0000223, 0.0009516, -0.0000095, 0.0009423, -0.0005054, 0.0005628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004886
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004757
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010744, -0.0005718, -0.0010949, -0.0005765, -0.0002732, 0.0003108
1: -0.0070371, -0.0057616, -0.0070890, -0.0057736, -0.0006934, 0.0007888
2: 0.0306642, 0.0314555, 0.0306320, 0.0314480, -0.0004302, 0.0004894
3: 0.0008271, 0.0023047, 0.0008411, 0.0023648, -0.0009138, 0.0008032
4: -0.0060509, -0.0047536, -0.0061037, -0.0047658, -0.0007053, 0.0008023
5: 0.0114463, 0.0119377, 0.0114263, 0.0119330, -0.0002671, 0.0003039
6: 0.0014379, 0.0033131, 0.0014556, 0.0033894, -0.0011597, 0.0010194
7: 0.9790654, 0.9803777, 0.9790778, 0.9804310, -0.0008115, 0.0007133
8: -0.0090094, -0.0076025, -0.0089961, -0.0075453, -0.0008701, 0.0007648
9: 0.0000223, 0.0009516, -0.0000155, 0.0009429, -0.0005052, 0.0005747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004886
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004758
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010818, -0.0005708, -0.0010924, -0.0005780, -0.0002825, 0.0003058
1: -0.0070558, -0.0057591, -0.0070827, -0.0057772, -0.0007168, 0.0007759
2: 0.0306525, 0.0314570, 0.0306359, 0.0314458, -0.0004447, 0.0004814
3: 0.0008243, 0.0023264, 0.0008452, 0.0023575, -0.0008989, 0.0008304
4: -0.0060700, -0.0047510, -0.0060973, -0.0047695, -0.0007291, 0.0007893
5: 0.0114390, 0.0119386, 0.0114287, 0.0119316, -0.0002762, 0.0002989
6: 0.0014342, 0.0033407, 0.0014608, 0.0033801, -0.0011408, 0.0010539
7: 0.9790629, 0.9803969, 0.9790815, 0.9804245, -0.0007983, 0.0007375
8: -0.0090121, -0.0075818, -0.0089922, -0.0075523, -0.0008559, 0.0007907
9: 0.0000086, 0.0009534, -0.0000109, 0.0009402, -0.0005223, 0.0005654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004758, upper bound: 0.0004957
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004758, upper bound: 0.0004832
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010818, -0.0005708, -0.0010990, -0.0005769, -0.0002718, 0.0003021
1: -0.0070558, -0.0057591, -0.0070993, -0.0057745, -0.0006896, 0.0007665
2: 0.0306525, 0.0314570, 0.0306256, 0.0314475, -0.0004278, 0.0004755
3: 0.0008243, 0.0023264, 0.0008420, 0.0023768, -0.0008880, 0.0007989
4: -0.0060700, -0.0047510, -0.0061142, -0.0047666, -0.0007015, 0.0007797
5: 0.0114390, 0.0119386, 0.0114223, 0.0119327, -0.0002657, 0.0002953
6: 0.0014342, 0.0033407, 0.0014568, 0.0034046, -0.0011269, 0.0010139
7: 0.9790629, 0.9803969, 0.9790787, 0.9804416, -0.0007886, 0.0007095
8: -0.0090121, -0.0075818, -0.0089952, -0.0075339, -0.0008455, 0.0007607
9: 0.0000086, 0.0009534, -0.0000231, 0.0009423, -0.0005025, 0.0005585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004758, upper bound: 0.0004969
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004758, upper bound: 0.0004843
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010744, -0.0005718, -0.0011049, -0.0005571, -0.0002888, 0.0003132
1: -0.0070371, -0.0057616, -0.0071144, -0.0057244, -0.0007329, 0.0007948
2: 0.0306642, 0.0314555, 0.0306162, 0.0314786, -0.0004547, 0.0004931
3: 0.0008271, 0.0023047, 0.0007840, 0.0023943, -0.0009208, 0.0008491
4: -0.0060509, -0.0047536, -0.0061296, -0.0047157, -0.0007455, 0.0008085
5: 0.0114463, 0.0119377, 0.0114165, 0.0119520, -0.0002824, 0.0003062
6: 0.0014379, 0.0033131, 0.0013831, 0.0034268, -0.0011686, 0.0010776
7: 0.9790654, 0.9803777, 0.9790272, 0.9804571, -0.0008177, 0.0007540
8: -0.0090094, -0.0076025, -0.0090505, -0.0075172, -0.0008767, 0.0008085
9: 0.0000223, 0.0009516, -0.0000341, 0.0009788, -0.0005340, 0.0005791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004722
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004628
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010744, -0.0005718, -0.0011088, -0.0005553, -0.0002888, 0.0003197
1: -0.0070371, -0.0057616, -0.0071243, -0.0057197, -0.0007328, 0.0008113
2: 0.0306642, 0.0314555, 0.0306101, 0.0314815, -0.0004546, 0.0005033
3: 0.0008271, 0.0023047, 0.0007786, 0.0024057, -0.0009398, 0.0008489
4: -0.0060509, -0.0047536, -0.0061396, -0.0047109, -0.0007454, 0.0008252
5: 0.0114463, 0.0119377, 0.0114127, 0.0119538, -0.0002823, 0.0003126
6: 0.0014379, 0.0033131, 0.0013762, 0.0034413, -0.0011928, 0.0010774
7: 0.9790654, 0.9803777, 0.9790223, 0.9804674, -0.0008346, 0.0007539
8: -0.0090094, -0.0076025, -0.0090557, -0.0075063, -0.0008949, 0.0008083
9: 0.0000223, 0.0009516, -0.0000412, 0.0009822, -0.0005339, 0.0005911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004723
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004630
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010818, -0.0005708, -0.0011045, -0.0005571, -0.0003000, 0.0003180
1: -0.0070558, -0.0057591, -0.0071135, -0.0057243, -0.0007613, 0.0008071
2: 0.0306525, 0.0314570, 0.0306168, 0.0314786, -0.0004723, 0.0005007
3: 0.0008243, 0.0023264, 0.0007839, 0.0023932, -0.0009350, 0.0008819
4: -0.0060700, -0.0047510, -0.0061286, -0.0047156, -0.0007744, 0.0008209
5: 0.0114390, 0.0119386, 0.0114168, 0.0119520, -0.0002933, 0.0003109
6: 0.0014342, 0.0033407, 0.0013830, 0.0034254, -0.0011866, 0.0011193
7: 0.9790629, 0.9803969, 0.9790271, 0.9804562, -0.0008303, 0.0007832
8: -0.0090121, -0.0075818, -0.0090506, -0.0075182, -0.0008902, 0.0008397
9: 0.0000086, 0.0009534, -0.0000334, 0.0009788, -0.0005547, 0.0005880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004750
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004650
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010818, -0.0005708, -0.0011140, -0.0005556, -0.0002885, 0.0003141
1: -0.0070558, -0.0057591, -0.0071376, -0.0057204, -0.0007320, 0.0007971
2: 0.0306525, 0.0314570, 0.0306018, 0.0314811, -0.0004541, 0.0004945
3: 0.0008243, 0.0023264, 0.0007794, 0.0024212, -0.0009235, 0.0008480
4: -0.0060700, -0.0047510, -0.0061532, -0.0047116, -0.0007446, 0.0008108
5: 0.0114390, 0.0119386, 0.0114075, 0.0119535, -0.0002820, 0.0003071
6: 0.0014342, 0.0033407, 0.0013773, 0.0034609, -0.0011720, 0.0010762
7: 0.9790629, 0.9803969, 0.9790230, 0.9804810, -0.0008201, 0.0007531
8: -0.0090121, -0.0075818, -0.0090549, -0.0074916, -0.0008793, 0.0008074
9: 0.0000086, 0.0009534, -0.0000510, 0.0009817, -0.0005334, 0.0005808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004766
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004668
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0005555, -0.0010769, -0.0005854, -0.0002767, 0.0003001
1: -0.0070578, -0.0057203, -0.0070434, -0.0057960, -0.0007022, 0.0007614
2: 0.0306513, 0.0314812, 0.0306603, 0.0314341, -0.0004356, 0.0004724
3: 0.0007792, 0.0023288, 0.0008670, 0.0023120, -0.0008821, 0.0008135
4: -0.0060720, -0.0047115, -0.0060573, -0.0047886, -0.0007143, 0.0007745
5: 0.0114383, 0.0119536, 0.0114438, 0.0119244, -0.0002705, 0.0002934
6: 0.0013771, 0.0033436, 0.0014885, 0.0033223, -0.0011195, 0.0010324
7: 0.9790228, 0.9803990, 0.9791008, 0.9803841, -0.0007834, 0.0007224
8: -0.0090550, -0.0075796, -0.0089714, -0.0075956, -0.0008399, 0.0007745
9: 0.0000072, 0.0009818, 0.0000177, 0.0009265, -0.0005116, 0.0005548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004875
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004875
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0005555, -0.0010809, -0.0005854, -0.0002745, 0.0003019
1: -0.0070578, -0.0057203, -0.0070535, -0.0057961, -0.0006966, 0.0007662
2: 0.0306513, 0.0314812, 0.0306540, 0.0314341, -0.0004322, 0.0004753
3: 0.0007792, 0.0023288, 0.0008671, 0.0023238, -0.0008876, 0.0008070
4: -0.0060720, -0.0047115, -0.0060677, -0.0047886, -0.0007086, 0.0007793
5: 0.0114383, 0.0119536, 0.0114399, 0.0119244, -0.0002684, 0.0002952
6: 0.0013771, 0.0033436, 0.0014886, 0.0033373, -0.0011265, 0.0010242
7: 0.9790228, 0.9803990, 0.9791009, 0.9803945, -0.0007882, 0.0007167
8: -0.0090550, -0.0075796, -0.0089714, -0.0075844, -0.0008451, 0.0007684
9: 0.0000072, 0.0009818, 0.0000103, 0.0009265, -0.0005076, 0.0005582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004875
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004875
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0005532, -0.0010767, -0.0005874, -0.0002825, 0.0002973
1: -0.0070819, -0.0057145, -0.0070428, -0.0058013, -0.0007169, 0.0007545
2: 0.0306364, 0.0314847, 0.0306607, 0.0314309, -0.0004448, 0.0004681
3: 0.0007726, 0.0023567, 0.0008731, 0.0023113, -0.0008741, 0.0008305
4: -0.0060965, -0.0047057, -0.0060567, -0.0047939, -0.0007292, 0.0007675
5: 0.0114290, 0.0119558, 0.0114441, 0.0119224, -0.0002762, 0.0002907
6: 0.0013686, 0.0033790, 0.0014962, 0.0033215, -0.0011093, 0.0010540
7: 0.9790170, 0.9804238, 0.9791062, 0.9803835, -0.0007763, 0.0007376
8: -0.0090613, -0.0075530, -0.0089657, -0.0075962, -0.0008323, 0.0007908
9: -0.0000104, 0.0009859, 0.0000181, 0.0009227, -0.0005224, 0.0005498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004651, upper bound: 0.0004969
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004651, upper bound: 0.0004969
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0005532, -0.0010836, -0.0005857, -0.0002727, 0.0002971
1: -0.0070819, -0.0057145, -0.0070604, -0.0057968, -0.0006919, 0.0007539
2: 0.0306364, 0.0314847, 0.0306497, 0.0314337, -0.0004293, 0.0004677
3: 0.0007726, 0.0023567, 0.0008679, 0.0023317, -0.0008734, 0.0008016
4: -0.0060965, -0.0047057, -0.0060747, -0.0047893, -0.0007038, 0.0007669
5: 0.0114290, 0.0119558, 0.0114373, 0.0119241, -0.0002666, 0.0002905
6: 0.0013686, 0.0033790, 0.0014896, 0.0033474, -0.0011085, 0.0010173
7: 0.9790170, 0.9804238, 0.9791017, 0.9804016, -0.0007756, 0.0007118
8: -0.0090613, -0.0075530, -0.0089706, -0.0075768, -0.0008316, 0.0007632
9: -0.0000104, 0.0009859, 0.0000053, 0.0009260, -0.0005041, 0.0005493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004651, upper bound: 0.0004981
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004651, upper bound: 0.0004982
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0005555, -0.0010908, -0.0005657, -0.0002771, 0.0002978
1: -0.0070578, -0.0057203, -0.0070786, -0.0057461, -0.0007031, 0.0007556
2: 0.0306513, 0.0314812, 0.0306384, 0.0314651, -0.0004362, 0.0004688
3: 0.0007792, 0.0023288, 0.0008092, 0.0023528, -0.0008753, 0.0008145
4: -0.0060720, -0.0047115, -0.0060932, -0.0047378, -0.0007151, 0.0007686
5: 0.0114383, 0.0119536, 0.0114303, 0.0119436, -0.0002709, 0.0002911
6: 0.0013771, 0.0033436, 0.0014151, 0.0033742, -0.0011109, 0.0010337
7: 0.9790228, 0.9803990, 0.9790494, 0.9804203, -0.0007774, 0.0007233
8: -0.0090550, -0.0075796, -0.0090265, -0.0075567, -0.0008335, 0.0007755
9: 0.0000072, 0.0009818, -0.0000080, 0.0009629, -0.0005123, 0.0005505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004669, upper bound: 0.0004876
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004669, upper bound: 0.0004876
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010826, -0.0005555, -0.0010967, -0.0005627, -0.0002760, 0.0003008
1: -0.0070578, -0.0057203, -0.0070936, -0.0057386, -0.0007003, 0.0007632
2: 0.0306513, 0.0314812, 0.0306291, 0.0314698, -0.0004345, 0.0004735
3: 0.0007792, 0.0023288, 0.0008004, 0.0023702, -0.0008841, 0.0008112
4: -0.0060720, -0.0047115, -0.0061084, -0.0047301, -0.0007123, 0.0007763
5: 0.0114383, 0.0119536, 0.0114245, 0.0119465, -0.0002698, 0.0002940
6: 0.0013771, 0.0033436, 0.0014040, 0.0033962, -0.0011221, 0.0010296
7: 0.9790228, 0.9803990, 0.9790418, 0.9804357, -0.0007852, 0.0007204
8: -0.0090550, -0.0075796, -0.0090348, -0.0075402, -0.0008418, 0.0007724
9: 0.0000072, 0.0009818, -0.0000189, 0.0009684, -0.0005102, 0.0005561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004669, upper bound: 0.0004876
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004669, upper bound: 0.0004876
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0005532, -0.0010897, -0.0005659, -0.0002876, 0.0002959
1: -0.0070819, -0.0057145, -0.0070759, -0.0057465, -0.0007297, 0.0007509
2: 0.0306364, 0.0314847, 0.0306401, 0.0314649, -0.0004527, 0.0004659
3: 0.0007726, 0.0023567, 0.0008096, 0.0023496, -0.0008699, 0.0008453
4: -0.0060965, -0.0047057, -0.0060904, -0.0047382, -0.0007422, 0.0007638
5: 0.0114290, 0.0119558, 0.0114313, 0.0119435, -0.0002811, 0.0002893
6: 0.0013686, 0.0033790, 0.0014156, 0.0033701, -0.0011040, 0.0010728
7: 0.9790170, 0.9804238, 0.9790499, 0.9804175, -0.0007725, 0.0007507
8: -0.0090613, -0.0075530, -0.0090261, -0.0075598, -0.0008283, 0.0008049
9: -0.0000104, 0.0009859, -0.0000059, 0.0009626, -0.0005317, 0.0005471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004655, upper bound: 0.0004970
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004655, upper bound: 0.0004970
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010921, -0.0005532, -0.0010991, -0.0005630, -0.0002766, 0.0002962
1: -0.0070819, -0.0057145, -0.0070998, -0.0057392, -0.0007020, 0.0007515
2: 0.0306364, 0.0314847, 0.0306253, 0.0314694, -0.0004355, 0.0004663
3: 0.0007726, 0.0023567, 0.0008011, 0.0023774, -0.0008706, 0.0008132
4: -0.0060965, -0.0047057, -0.0061147, -0.0047307, -0.0007140, 0.0007644
5: 0.0114290, 0.0119558, 0.0114221, 0.0119463, -0.0002704, 0.0002895
6: 0.0013686, 0.0033790, 0.0014049, 0.0034053, -0.0011049, 0.0010320
7: 0.9790170, 0.9804238, 0.9790424, 0.9804422, -0.0007732, 0.0007222
8: -0.0090613, -0.0075530, -0.0090341, -0.0075333, -0.0008290, 0.0007743
9: -0.0000104, 0.0009859, -0.0000234, 0.0009680, -0.0005115, 0.0005476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004654, upper bound: 0.0004982
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004654, upper bound: 0.0004982
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010890, -0.0005531, -0.0010769, -0.0005854, -0.0002814, 0.0003023
1: -0.0070740, -0.0057142, -0.0070434, -0.0057960, -0.0007141, 0.0007671
2: 0.0306413, 0.0314849, 0.0306603, 0.0314341, -0.0004430, 0.0004759
3: 0.0007722, 0.0023474, 0.0008670, 0.0023120, -0.0008887, 0.0008272
4: -0.0060884, -0.0047053, -0.0060573, -0.0047886, -0.0007263, 0.0007803
5: 0.0114320, 0.0119559, 0.0114438, 0.0119244, -0.0002751, 0.0002956
6: 0.0013681, 0.0033673, 0.0014885, 0.0033223, -0.0011278, 0.0010499
7: 0.9790166, 0.9804156, 0.9791008, 0.9803841, -0.0007892, 0.0007346
8: -0.0090617, -0.0075618, -0.0089714, -0.0075956, -0.0008462, 0.0007877
9: -0.0000046, 0.0009862, 0.0000177, 0.0009265, -0.0005203, 0.0005589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004921
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004921
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010890, -0.0005531, -0.0010809, -0.0005854, -0.0002795, 0.0003030
1: -0.0070740, -0.0057142, -0.0070535, -0.0057961, -0.0007091, 0.0007690
2: 0.0306413, 0.0314849, 0.0306540, 0.0314341, -0.0004400, 0.0004771
3: 0.0007722, 0.0023474, 0.0008671, 0.0023238, -0.0008908, 0.0008215
4: -0.0060884, -0.0047053, -0.0060677, -0.0047886, -0.0007213, 0.0007822
5: 0.0114320, 0.0119559, 0.0114399, 0.0119244, -0.0002732, 0.0002963
6: 0.0013681, 0.0033673, 0.0014886, 0.0033373, -0.0011306, 0.0010426
7: 0.9790166, 0.9804156, 0.9791009, 0.9803945, -0.0007911, 0.0007296
8: -0.0090617, -0.0075618, -0.0089714, -0.0075844, -0.0008482, 0.0007822
9: -0.0000046, 0.0009862, 0.0000103, 0.0009265, -0.0005167, 0.0005603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004923
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004923
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010977, -0.0005508, -0.0010800, -0.0005853, -0.0002819, 0.0003070
1: -0.0070962, -0.0057082, -0.0070512, -0.0057959, -0.0007153, 0.0007791
2: 0.0306275, 0.0314886, 0.0306554, 0.0314342, -0.0004438, 0.0004834
3: 0.0007653, 0.0023732, 0.0008668, 0.0023210, -0.0009025, 0.0008287
4: -0.0061111, -0.0046993, -0.0060653, -0.0047884, -0.0007276, 0.0007925
5: 0.0114235, 0.0119582, 0.0114408, 0.0119245, -0.0002756, 0.0003002
6: 0.0013594, 0.0034001, 0.0014882, 0.0033338, -0.0011455, 0.0010517
7: 0.9790106, 0.9804385, 0.9791006, 0.9803922, -0.0008015, 0.0007359
8: -0.0090683, -0.0075373, -0.0089716, -0.0075870, -0.0008594, 0.0007890
9: -0.0000208, 0.0009905, 0.0000120, 0.0009267, -0.0005212, 0.0005677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004680, upper bound: 0.0005028
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004680, upper bound: 0.0005028
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010977, -0.0005508, -0.0010842, -0.0005854, -0.0002792, 0.0003093
1: -0.0070962, -0.0057082, -0.0070618, -0.0057960, -0.0007086, 0.0007848
2: 0.0306275, 0.0314886, 0.0306489, 0.0314341, -0.0004396, 0.0004869
3: 0.0007653, 0.0023732, 0.0008670, 0.0023333, -0.0009092, 0.0008209
4: -0.0061111, -0.0046993, -0.0060760, -0.0047885, -0.0007208, 0.0007983
5: 0.0114235, 0.0119582, 0.0114367, 0.0119244, -0.0002730, 0.0003024
6: 0.0013594, 0.0034001, 0.0014884, 0.0033494, -0.0011539, 0.0010418
7: 0.9790106, 0.9804385, 0.9791008, 0.9804030, -0.0008074, 0.0007290
8: -0.0090683, -0.0075373, -0.0089715, -0.0075753, -0.0008657, 0.0007816
9: -0.0000208, 0.0009905, 0.0000043, 0.0009266, -0.0005163, 0.0005718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004680, upper bound: 0.0005029
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004680, upper bound: 0.0005029
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010890, -0.0005531, -0.0010908, -0.0005657, -0.0002811, 0.0002966
1: -0.0070740, -0.0057142, -0.0070786, -0.0057461, -0.0007134, 0.0007526
2: 0.0306413, 0.0314849, 0.0306384, 0.0314651, -0.0004426, 0.0004669
3: 0.0007722, 0.0023474, 0.0008092, 0.0023528, -0.0008718, 0.0008264
4: -0.0060884, -0.0047053, -0.0060932, -0.0047378, -0.0007256, 0.0007655
5: 0.0114320, 0.0119559, 0.0114303, 0.0119436, -0.0002749, 0.0002899
6: 0.0013681, 0.0033673, 0.0014151, 0.0033742, -0.0011065, 0.0010489
7: 0.9790166, 0.9804156, 0.9790494, 0.9804203, -0.0007742, 0.0007339
8: -0.0090617, -0.0075618, -0.0090265, -0.0075567, -0.0008301, 0.0007869
9: -0.0000046, 0.0009862, -0.0000080, 0.0009629, -0.0005198, 0.0005483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004670, upper bound: 0.0004922
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004670, upper bound: 0.0004922
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010890, -0.0005531, -0.0010967, -0.0005627, -0.0002809, 0.0003016
1: -0.0070740, -0.0057142, -0.0070936, -0.0057386, -0.0007128, 0.0007654
2: 0.0306413, 0.0314849, 0.0306291, 0.0314698, -0.0004422, 0.0004748
3: 0.0007722, 0.0023474, 0.0008004, 0.0023702, -0.0008866, 0.0008258
4: -0.0060884, -0.0047053, -0.0061084, -0.0047301, -0.0007251, 0.0007785
5: 0.0114320, 0.0119559, 0.0114245, 0.0119465, -0.0002746, 0.0002949
6: 0.0013681, 0.0033673, 0.0014040, 0.0033962, -0.0011252, 0.0010480
7: 0.9790166, 0.9804156, 0.9790418, 0.9804357, -0.0007874, 0.0007333
8: -0.0090617, -0.0075618, -0.0090348, -0.0075402, -0.0008442, 0.0007863
9: -0.0000046, 0.0009862, -0.0000189, 0.0009684, -0.0005194, 0.0005576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004670, upper bound: 0.0004925
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004670, upper bound: 0.0004925
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010977, -0.0005508, -0.0010950, -0.0005656, -0.0002820, 0.0003041
1: -0.0070962, -0.0057082, -0.0070892, -0.0057459, -0.0007156, 0.0007717
2: 0.0306275, 0.0314886, 0.0306319, 0.0314653, -0.0004440, 0.0004788
3: 0.0007653, 0.0023732, 0.0008089, 0.0023651, -0.0008940, 0.0008290
4: -0.0061111, -0.0046993, -0.0061039, -0.0047375, -0.0007279, 0.0007850
5: 0.0114235, 0.0119582, 0.0114262, 0.0119437, -0.0002757, 0.0002973
6: 0.0013594, 0.0034001, 0.0014147, 0.0033897, -0.0011346, 0.0010521
7: 0.9790106, 0.9804385, 0.9790492, 0.9804312, -0.0007940, 0.0007362
8: -0.0090683, -0.0075373, -0.0090268, -0.0075451, -0.0008513, 0.0007893
9: -0.0000208, 0.0009905, -0.0000157, 0.0009631, -0.0005214, 0.0005623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004683, upper bound: 0.0005028
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004683, upper bound: 0.0005028
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010977, -0.0005508, -0.0011005, -0.0005627, -0.0002818, 0.0003090
1: -0.0070962, -0.0057082, -0.0071034, -0.0057384, -0.0007150, 0.0007841
2: 0.0306275, 0.0314886, 0.0306231, 0.0314699, -0.0004436, 0.0004865
3: 0.0007653, 0.0023732, 0.0008002, 0.0023815, -0.0009083, 0.0008283
4: -0.0061111, -0.0046993, -0.0061183, -0.0047299, -0.0007273, 0.0007976
5: 0.0114235, 0.0119582, 0.0114207, 0.0119466, -0.0002755, 0.0003021
6: 0.0013594, 0.0034001, 0.0014037, 0.0034105, -0.0011528, 0.0010512
7: 0.9790106, 0.9804385, 0.9790416, 0.9804457, -0.0008067, 0.0007356
8: -0.0090683, -0.0075373, -0.0090350, -0.0075294, -0.0008649, 0.0007887
9: -0.0000208, 0.0009905, -0.0000260, 0.0009685, -0.0005210, 0.0005713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004683, upper bound: 0.0005029
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004683, upper bound: 0.0005029
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010828, -0.0005531, -0.0010916, -0.0005768, -0.0002854, 0.0003245
1: -0.0070584, -0.0057142, -0.0070807, -0.0057744, -0.0007242, 0.0008234
2: 0.0306510, 0.0314849, 0.0306371, 0.0314476, -0.0004493, 0.0005108
3: 0.0007722, 0.0023294, 0.0008419, 0.0023552, -0.0009539, 0.0008390
4: -0.0060726, -0.0047053, -0.0060953, -0.0047666, -0.0007367, 0.0008375
5: 0.0114380, 0.0119559, 0.0114295, 0.0119327, -0.0002790, 0.0003172
6: 0.0013681, 0.0033444, 0.0014567, 0.0033772, -0.0012106, 0.0010648
7: 0.9790166, 0.9803995, 0.9790786, 0.9804224, -0.0008471, 0.0007451
8: -0.0090618, -0.0075790, -0.0089953, -0.0075545, -0.0009082, 0.0007989
9: 0.0000068, 0.0009862, -0.0000095, 0.0009423, -0.0005277, 0.0005999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004647, upper bound: 0.0004998
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004647, upper bound: 0.0004875
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010828, -0.0005531, -0.0010949, -0.0005765, -0.0002832, 0.0003253
1: -0.0070584, -0.0057142, -0.0070890, -0.0057736, -0.0007186, 0.0008255
2: 0.0306510, 0.0314849, 0.0306320, 0.0314480, -0.0004459, 0.0005121
3: 0.0007722, 0.0023294, 0.0008411, 0.0023648, -0.0009563, 0.0008325
4: -0.0060726, -0.0047053, -0.0061037, -0.0047658, -0.0007310, 0.0008397
5: 0.0114380, 0.0119559, 0.0114263, 0.0119330, -0.0002769, 0.0003180
6: 0.0013681, 0.0033444, 0.0014556, 0.0033894, -0.0012137, 0.0010566
7: 0.9790166, 0.9803995, 0.9790778, 0.9804310, -0.0008493, 0.0007393
8: -0.0090618, -0.0075790, -0.0089961, -0.0075453, -0.0009106, 0.0007927
9: 0.0000068, 0.0009862, -0.0000155, 0.0009429, -0.0005236, 0.0006015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004647, upper bound: 0.0004998
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004647, upper bound: 0.0004875
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010923, -0.0005515, -0.0010913, -0.0005780, -0.0002891, 0.0003222
1: -0.0070825, -0.0057101, -0.0070798, -0.0057774, -0.0007337, 0.0008177
2: 0.0306360, 0.0314875, 0.0306377, 0.0314457, -0.0004552, 0.0005073
3: 0.0007674, 0.0023573, 0.0008454, 0.0023542, -0.0009473, 0.0008500
4: -0.0060971, -0.0047011, -0.0060944, -0.0047696, -0.0007463, 0.0008317
5: 0.0114288, 0.0119575, 0.0114298, 0.0119316, -0.0002827, 0.0003150
6: 0.0013621, 0.0033798, 0.0014610, 0.0033760, -0.0012022, 0.0010787
7: 0.9790124, 0.9804243, 0.9790816, 0.9804216, -0.0008412, 0.0007549
8: -0.0090663, -0.0075525, -0.0089920, -0.0075554, -0.0009019, 0.0008093
9: -0.0000108, 0.0009892, -0.0000089, 0.0009401, -0.0005346, 0.0005958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004628, upper bound: 0.0005106
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004628, upper bound: 0.0004969
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010923, -0.0005515, -0.0010981, -0.0005769, -0.0002799, 0.0003211
1: -0.0070825, -0.0057101, -0.0070972, -0.0057745, -0.0007102, 0.0008148
2: 0.0306360, 0.0314875, 0.0306269, 0.0314475, -0.0004406, 0.0005055
3: 0.0007674, 0.0023573, 0.0008421, 0.0023744, -0.0009439, 0.0008228
4: -0.0060971, -0.0047011, -0.0061121, -0.0047667, -0.0007224, 0.0008287
5: 0.0114288, 0.0119575, 0.0114231, 0.0119327, -0.0002736, 0.0003139
6: 0.0013621, 0.0033798, 0.0014568, 0.0034015, -0.0011979, 0.0010442
7: 0.9790124, 0.9804243, 0.9790788, 0.9804395, -0.0008382, 0.0007307
8: -0.0090663, -0.0075525, -0.0089952, -0.0075362, -0.0008987, 0.0007834
9: -0.0000108, 0.0009892, -0.0000215, 0.0009422, -0.0005175, 0.0005936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004628, upper bound: 0.0005120
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004628, upper bound: 0.0004982
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010828, -0.0005531, -0.0011049, -0.0005571, -0.0002850, 0.0003212
1: -0.0070584, -0.0057142, -0.0071144, -0.0057244, -0.0007233, 0.0008151
2: 0.0306510, 0.0314849, 0.0306162, 0.0314786, -0.0004487, 0.0005057
3: 0.0007722, 0.0023294, 0.0007840, 0.0023943, -0.0009442, 0.0008379
4: -0.0060726, -0.0047053, -0.0061296, -0.0047157, -0.0007357, 0.0008291
5: 0.0114380, 0.0119559, 0.0114165, 0.0119520, -0.0002787, 0.0003140
6: 0.0013681, 0.0033444, 0.0013831, 0.0034268, -0.0011983, 0.0010634
7: 0.9790166, 0.9803995, 0.9790272, 0.9804571, -0.0008385, 0.0007441
8: -0.0090618, -0.0075790, -0.0090505, -0.0075172, -0.0008990, 0.0007978
9: 0.0000068, 0.0009862, -0.0000341, 0.0009788, -0.0005270, 0.0005939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004652, upper bound: 0.0004999
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004652, upper bound: 0.0004876
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010828, -0.0005531, -0.0011088, -0.0005553, -0.0002842, 0.0003242
1: -0.0070584, -0.0057142, -0.0071243, -0.0057197, -0.0007212, 0.0008226
2: 0.0306510, 0.0314849, 0.0306101, 0.0314815, -0.0004474, 0.0005104
3: 0.0007722, 0.0023294, 0.0007786, 0.0024057, -0.0009530, 0.0008354
4: -0.0060726, -0.0047053, -0.0061396, -0.0047109, -0.0007335, 0.0008368
5: 0.0114380, 0.0119559, 0.0114127, 0.0119538, -0.0002778, 0.0003169
6: 0.0013681, 0.0033444, 0.0013762, 0.0034413, -0.0012095, 0.0010603
7: 0.9790166, 0.9803995, 0.9790223, 0.9804674, -0.0008463, 0.0007419
8: -0.0090618, -0.0075790, -0.0090557, -0.0075063, -0.0009074, 0.0007955
9: 0.0000068, 0.0009862, -0.0000412, 0.0009822, -0.0005254, 0.0005994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004652, upper bound: 0.0004999
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004652, upper bound: 0.0004876
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010923, -0.0005515, -0.0011033, -0.0005572, -0.0002940, 0.0003208
1: -0.0070825, -0.0057101, -0.0071104, -0.0057244, -0.0007461, 0.0008141
2: 0.0306360, 0.0314875, 0.0306187, 0.0314786, -0.0004629, 0.0005051
3: 0.0007674, 0.0023573, 0.0007841, 0.0023896, -0.0009431, 0.0008643
4: -0.0060971, -0.0047011, -0.0061255, -0.0047157, -0.0007589, 0.0008281
5: 0.0114288, 0.0119575, 0.0114180, 0.0119520, -0.0002875, 0.0003136
6: 0.0013621, 0.0033798, 0.0013832, 0.0034209, -0.0011969, 0.0010970
7: 0.9790124, 0.9804243, 0.9790272, 0.9804531, -0.0008375, 0.0007676
8: -0.0090663, -0.0075525, -0.0090504, -0.0075217, -0.0008980, 0.0008230
9: -0.0000108, 0.0009892, -0.0000311, 0.0009787, -0.0005436, 0.0005932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004632, upper bound: 0.0005108
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004632, upper bound: 0.0004970
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010923, -0.0005515, -0.0011131, -0.0005556, -0.0002829, 0.0003197
1: -0.0070825, -0.0057101, -0.0071353, -0.0057205, -0.0007179, 0.0008114
2: 0.0306360, 0.0314875, 0.0306033, 0.0314810, -0.0004454, 0.0005034
3: 0.0007674, 0.0023573, 0.0007795, 0.0024185, -0.0009399, 0.0008317
4: -0.0060971, -0.0047011, -0.0061508, -0.0047117, -0.0007303, 0.0008253
5: 0.0114288, 0.0119575, 0.0114084, 0.0119535, -0.0002766, 0.0003126
6: 0.0013621, 0.0033798, 0.0013774, 0.0034575, -0.0011929, 0.0010555
7: 0.9790124, 0.9804243, 0.9790231, 0.9804786, -0.0008347, 0.0007386
8: -0.0090663, -0.0075525, -0.0090548, -0.0074942, -0.0008950, 0.0007919
9: -0.0000108, 0.0009892, -0.0000492, 0.0009816, -0.0005231, 0.0005912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004632, upper bound: 0.0005121
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004632, upper bound: 0.0004982
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010892, -0.0005510, -0.0010916, -0.0005768, -0.0002901, 0.0003265
1: -0.0070746, -0.0057089, -0.0070807, -0.0057744, -0.0007363, 0.0008286
2: 0.0306409, 0.0314882, 0.0306371, 0.0314476, -0.0004568, 0.0005141
3: 0.0007661, 0.0023481, 0.0008419, 0.0023552, -0.0009599, 0.0008530
4: -0.0060890, -0.0046999, -0.0060953, -0.0047666, -0.0007489, 0.0008428
5: 0.0114318, 0.0119580, 0.0114295, 0.0119327, -0.0002837, 0.0003192
6: 0.0013604, 0.0033682, 0.0014567, 0.0033772, -0.0012182, 0.0010825
7: 0.9790112, 0.9804161, 0.9790786, 0.9804224, -0.0008525, 0.0007575
8: -0.0090676, -0.0075612, -0.0089953, -0.0075545, -0.0009140, 0.0008121
9: -0.0000050, 0.0009900, -0.0000095, 0.0009423, -0.0005365, 0.0006037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004648, upper bound: 0.0005041
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004648, upper bound: 0.0004921
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010892, -0.0005510, -0.0010949, -0.0005765, -0.0002879, 0.0003273
1: -0.0070746, -0.0057089, -0.0070890, -0.0057736, -0.0007306, 0.0008305
2: 0.0306409, 0.0314882, 0.0306320, 0.0314480, -0.0004533, 0.0005153
3: 0.0007661, 0.0023481, 0.0008411, 0.0023648, -0.0009621, 0.0008463
4: -0.0060890, -0.0046999, -0.0061037, -0.0047658, -0.0007431, 0.0008448
5: 0.0114318, 0.0119580, 0.0114263, 0.0119330, -0.0002815, 0.0003200
6: 0.0013604, 0.0033682, 0.0014556, 0.0033894, -0.0012210, 0.0010741
7: 0.9790112, 0.9804161, 0.9790778, 0.9804310, -0.0008544, 0.0007516
8: -0.0090676, -0.0075612, -0.0089961, -0.0075453, -0.0009161, 0.0008058
9: -0.0000050, 0.0009900, -0.0000155, 0.0009429, -0.0005323, 0.0006051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004648, upper bound: 0.0005042
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004648, upper bound: 0.0004923
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0005493, -0.0010924, -0.0005780, -0.0002956, 0.0003233
1: -0.0070969, -0.0057045, -0.0070827, -0.0057772, -0.0007502, 0.0008203
2: 0.0306271, 0.0314909, 0.0306359, 0.0314458, -0.0004654, 0.0005089
3: 0.0007610, 0.0023740, 0.0008452, 0.0023575, -0.0009503, 0.0008691
4: -0.0061117, -0.0046954, -0.0060973, -0.0047695, -0.0007631, 0.0008344
5: 0.0114232, 0.0119597, 0.0114287, 0.0119316, -0.0002890, 0.0003160
6: 0.0013539, 0.0034010, 0.0014608, 0.0033801, -0.0012060, 0.0011030
7: 0.9790066, 0.9804391, 0.9790815, 0.9804245, -0.0008439, 0.0007718
8: -0.0090724, -0.0075366, -0.0089922, -0.0075523, -0.0009048, 0.0008275
9: -0.0000213, 0.0009932, -0.0000109, 0.0009402, -0.0005466, 0.0005977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004630, upper bound: 0.0005162
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004630, upper bound: 0.0005015
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0005493, -0.0010990, -0.0005769, -0.0002864, 0.0003231
1: -0.0070969, -0.0057045, -0.0070993, -0.0057745, -0.0007268, 0.0008199
2: 0.0306271, 0.0314909, 0.0306256, 0.0314475, -0.0004509, 0.0005087
3: 0.0007610, 0.0023740, 0.0008420, 0.0023768, -0.0009499, 0.0008420
4: -0.0061117, -0.0046954, -0.0061142, -0.0047666, -0.0007393, 0.0008340
5: 0.0114232, 0.0119597, 0.0114223, 0.0119327, -0.0002800, 0.0003159
6: 0.0013539, 0.0034010, 0.0014568, 0.0034046, -0.0012055, 0.0010685
7: 0.9790066, 0.9804391, 0.9790787, 0.9804416, -0.0008436, 0.0007477
8: -0.0090724, -0.0075366, -0.0089952, -0.0075339, -0.0009044, 0.0008017
9: -0.0000213, 0.0009932, -0.0000231, 0.0009423, -0.0005295, 0.0005974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004630, upper bound: 0.0005176
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004630, upper bound: 0.0005029
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010892, -0.0005510, -0.0011049, -0.0005571, -0.0002892, 0.0003194
1: -0.0070746, -0.0057089, -0.0071144, -0.0057244, -0.0007338, 0.0008105
2: 0.0306409, 0.0314882, 0.0306162, 0.0314786, -0.0004553, 0.0005028
3: 0.0007661, 0.0023481, 0.0007840, 0.0023943, -0.0009389, 0.0008501
4: -0.0060890, -0.0046999, -0.0061296, -0.0047157, -0.0007464, 0.0008244
5: 0.0114318, 0.0119580, 0.0114165, 0.0119520, -0.0002827, 0.0003123
6: 0.0013604, 0.0033682, 0.0013831, 0.0034268, -0.0011916, 0.0010788
7: 0.9790112, 0.9804161, 0.9790272, 0.9804571, -0.0008339, 0.0007549
8: -0.0090676, -0.0075612, -0.0090505, -0.0075172, -0.0008940, 0.0008094
9: -0.0000050, 0.0009900, -0.0000341, 0.0009788, -0.0005347, 0.0005906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004653, upper bound: 0.0005042
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004653, upper bound: 0.0004922
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010892, -0.0005510, -0.0011088, -0.0005553, -0.0002884, 0.0003249
1: -0.0070746, -0.0057089, -0.0071243, -0.0057197, -0.0007318, 0.0008244
2: 0.0306409, 0.0314882, 0.0306101, 0.0314815, -0.0004540, 0.0005115
3: 0.0007661, 0.0023481, 0.0007786, 0.0024057, -0.0009551, 0.0008477
4: -0.0060890, -0.0046999, -0.0061396, -0.0047109, -0.0007444, 0.0008386
5: 0.0114318, 0.0119580, 0.0114127, 0.0119538, -0.0002819, 0.0003176
6: 0.0013604, 0.0033682, 0.0013762, 0.0034413, -0.0012121, 0.0010759
7: 0.9790112, 0.9804161, 0.9790223, 0.9804674, -0.0008482, 0.0007529
8: -0.0090676, -0.0075612, -0.0090557, -0.0075063, -0.0009094, 0.0008072
9: -0.0000050, 0.0009900, -0.0000412, 0.0009822, -0.0005332, 0.0006007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004653, upper bound: 0.0005043
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004653, upper bound: 0.0004926
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0005493, -0.0011045, -0.0005571, -0.0002988, 0.0003204
1: -0.0070969, -0.0057045, -0.0071135, -0.0057243, -0.0007582, 0.0008131
2: 0.0306271, 0.0314909, 0.0306168, 0.0314786, -0.0004704, 0.0005045
3: 0.0007610, 0.0023740, 0.0007839, 0.0023932, -0.0009420, 0.0008784
4: -0.0061117, -0.0046954, -0.0061286, -0.0047156, -0.0007712, 0.0008271
5: 0.0114232, 0.0119597, 0.0114168, 0.0119520, -0.0002921, 0.0003133
6: 0.0013539, 0.0034010, 0.0013830, 0.0034254, -0.0011955, 0.0011148
7: 0.9790066, 0.9804391, 0.9790271, 0.9804562, -0.0008365, 0.0007801
8: -0.0090724, -0.0075366, -0.0090506, -0.0075182, -0.0008969, 0.0008364
9: -0.0000213, 0.0009932, -0.0000334, 0.0009788, -0.0005525, 0.0005924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004637, upper bound: 0.0005165
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004637, upper bound: 0.0005019
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010980, -0.0005493, -0.0011140, -0.0005556, -0.0002884, 0.0003183
1: -0.0070969, -0.0057045, -0.0071376, -0.0057204, -0.0007320, 0.0008078
2: 0.0306271, 0.0314909, 0.0306018, 0.0314811, -0.0004541, 0.0005012
3: 0.0007610, 0.0023740, 0.0007794, 0.0024212, -0.0009358, 0.0008480
4: -0.0061117, -0.0046954, -0.0061532, -0.0047116, -0.0007445, 0.0008217
5: 0.0114232, 0.0119597, 0.0114075, 0.0119535, -0.0002820, 0.0003112
6: 0.0013539, 0.0034010, 0.0013773, 0.0034609, -0.0011877, 0.0010762
7: 0.9790066, 0.9804391, 0.9790230, 0.9804810, -0.0008311, 0.0007531
8: -0.0090724, -0.0075366, -0.0090549, -0.0074916, -0.0008911, 0.0008074
9: -0.0000213, 0.0009932, -0.0000510, 0.0009817, -0.0005333, 0.0005886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004637, upper bound: 0.0005177
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004637, upper bound: 0.0005029
time: 0.66 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.54 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004727
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004727
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004727
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004727
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004823, upper bound: 0.0004794
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004823, upper bound: 0.0004794
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004823, upper bound: 0.0004804
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004823, upper bound: 0.0004804
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004637
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004637
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004637
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004637
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004974, upper bound: 0.0004655
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004974, upper bound: 0.0004655
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004974, upper bound: 0.0004670
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004974, upper bound: 0.0004670
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004757
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004757
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004758
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004866, upper bound: 0.0004758
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004843
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004843
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004843
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004843
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0004628
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0004628
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0004630
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005044, upper bound: 0.0004630
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004662
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004662
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004667
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0005056, upper bound: 0.0004667
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004861
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004727
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004861
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004727
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004757, upper bound: 0.0004932
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004757, upper bound: 0.0004794
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004757, upper bound: 0.0004945
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004757, upper bound: 0.0004804
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004721
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004637
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004721
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004637
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004921, upper bound: 0.0004755
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004921, upper bound: 0.0004655
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004921, upper bound: 0.0004770
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004921, upper bound: 0.0004670
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004886
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004757
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004886
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004794, upper bound: 0.0004758
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004758, upper bound: 0.0004957
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004758, upper bound: 0.0004832
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004758, upper bound: 0.0004969
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004758, upper bound: 0.0004843
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004722
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004628
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004723
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004969, upper bound: 0.0004630
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004750
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004650
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004766
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004923, upper bound: 0.0004668
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004875
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004875
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004875
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004875
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004651, upper bound: 0.0004969
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004651, upper bound: 0.0004969
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004651, upper bound: 0.0004981
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004651, upper bound: 0.0004982
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004669, upper bound: 0.0004876
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004669, upper bound: 0.0004876
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004669, upper bound: 0.0004876
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004669, upper bound: 0.0004876
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004655, upper bound: 0.0004970
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004655, upper bound: 0.0004970
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004654, upper bound: 0.0004982
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004654, upper bound: 0.0004982
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004921
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004921
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004923
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004665, upper bound: 0.0004923
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004680, upper bound: 0.0005028
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004680, upper bound: 0.0005028
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004680, upper bound: 0.0005029
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004680, upper bound: 0.0005029
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004670, upper bound: 0.0004922
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004670, upper bound: 0.0004922
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004670, upper bound: 0.0004925
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004670, upper bound: 0.0004925
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004683, upper bound: 0.0005028
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004683, upper bound: 0.0005028
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004683, upper bound: 0.0005029
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004683, upper bound: 0.0005029
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004647, upper bound: 0.0004998
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004647, upper bound: 0.0004875
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004647, upper bound: 0.0004998
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004647, upper bound: 0.0004875
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004628, upper bound: 0.0005106
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004628, upper bound: 0.0004969
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004628, upper bound: 0.0005120
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004628, upper bound: 0.0004982
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004652, upper bound: 0.0004999
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004652, upper bound: 0.0004876
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004652, upper bound: 0.0004999
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004652, upper bound: 0.0004876
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004632, upper bound: 0.0005108
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004632, upper bound: 0.0004970
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004632, upper bound: 0.0005121
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004632, upper bound: 0.0004982
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004648, upper bound: 0.0005041
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004648, upper bound: 0.0004921
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004648, upper bound: 0.0005042
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004648, upper bound: 0.0004923
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004630, upper bound: 0.0005162
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004630, upper bound: 0.0005015
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004630, upper bound: 0.0005176
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004630, upper bound: 0.0005029
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004653, upper bound: 0.0005042
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004653, upper bound: 0.0004922
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004653, upper bound: 0.0005043
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004653, upper bound: 0.0004926
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004637, upper bound: 0.0005165
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004637, upper bound: 0.0005019
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004637, upper bound: 0.0005177
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.54
Output dim: 7, lower bound: -0.0004637, upper bound: 0.0005029

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010686, -0.0005882, -0.0010769, -0.0005854, -0.0002617, 0.0002698
1: -0.0070223, -0.0058031, -0.0070434, -0.0057960, -0.0006642, 0.0006845
2: 0.0306734, 0.0314297, 0.0306603, 0.0314341, -0.0004121, 0.0004247
3: 0.0008752, 0.0022875, 0.0008670, 0.0023120, -0.0007930, 0.0007694
4: -0.0060359, -0.0047958, -0.0060573, -0.0047886, -0.0006756, 0.0006963
5: 0.0114520, 0.0119217, 0.0114438, 0.0119244, -0.0002559, 0.0002637
6: 0.0014989, 0.0032913, 0.0014885, 0.0033223, -0.0010064, 0.0009765
7: 0.9791081, 0.9803624, 0.9791008, 0.9803841, -0.0007042, 0.0006833
8: -0.0089636, -0.0076189, -0.0089714, -0.0075956, -0.0007551, 0.0007326
9: 0.0000331, 0.0009214, 0.0000177, 0.0009265, -0.0004839, 0.0004988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004819, upper bound: 0.0004728
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004819, upper bound: 0.0004728
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010834, -0.0005782, -0.0010769, -0.0005854, -0.0002838, 0.0002815
1: -0.0070599, -0.0057779, -0.0070434, -0.0057960, -0.0007202, 0.0007143
2: 0.0306500, 0.0314454, 0.0306603, 0.0314341, -0.0004468, 0.0004432
3: 0.0008460, 0.0023311, 0.0008670, 0.0023120, -0.0008275, 0.0008343
4: -0.0060741, -0.0047701, -0.0060573, -0.0047886, -0.0007325, 0.0007266
5: 0.0114375, 0.0119314, 0.0114438, 0.0119244, -0.0002775, 0.0002752
6: 0.0014618, 0.0033466, 0.0014885, 0.0033223, -0.0010502, 0.0010588
7: 0.9790821, 0.9804011, 0.9791008, 0.9803841, -0.0007349, 0.0007409
8: -0.0089914, -0.0075774, -0.0089714, -0.0075956, -0.0007879, 0.0007944
9: 0.0000057, 0.0009398, 0.0000177, 0.0009265, -0.0005247, 0.0005205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004819, upper bound: 0.0004728
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004819, upper bound: 0.0004729
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010686, -0.0005882, -0.0010809, -0.0005854, -0.0002595, 0.0002716
1: -0.0070223, -0.0058031, -0.0070535, -0.0057961, -0.0006586, 0.0006893
2: 0.0306734, 0.0314297, 0.0306540, 0.0314341, -0.0004086, 0.0004276
3: 0.0008752, 0.0022875, 0.0008671, 0.0023238, -0.0007985, 0.0007629
4: -0.0060359, -0.0047958, -0.0060677, -0.0047886, -0.0006699, 0.0007011
5: 0.0114520, 0.0119217, 0.0114399, 0.0119244, -0.0002537, 0.0002656
6: 0.0014989, 0.0032913, 0.0014886, 0.0033373, -0.0010134, 0.0009683
7: 0.9791081, 0.9803624, 0.9791009, 0.9803945, -0.0007091, 0.0006775
8: -0.0089636, -0.0076189, -0.0089714, -0.0075844, -0.0007603, 0.0007264
9: 0.0000331, 0.0009214, 0.0000103, 0.0009265, -0.0004799, 0.0005022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004854, upper bound: 0.0004727
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004854, upper bound: 0.0004727
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010834, -0.0005782, -0.0010809, -0.0005854, -0.0002816, 0.0002834
1: -0.0070599, -0.0057779, -0.0070535, -0.0057961, -0.0007146, 0.0007191
2: 0.0306500, 0.0314454, 0.0306540, 0.0314341, -0.0004433, 0.0004461
3: 0.0008460, 0.0023311, 0.0008671, 0.0023238, -0.0008330, 0.0008278
4: -0.0060741, -0.0047701, -0.0060677, -0.0047886, -0.0007269, 0.0007314
5: 0.0114375, 0.0119314, 0.0114399, 0.0119244, -0.0002753, 0.0002770
6: 0.0014618, 0.0033466, 0.0014886, 0.0033373, -0.0010572, 0.0010506
7: 0.9790821, 0.9804011, 0.9791009, 0.9803945, -0.0007398, 0.0007352
8: -0.0089914, -0.0075774, -0.0089714, -0.0075844, -0.0007931, 0.0007882
9: 0.0000057, 0.0009398, 0.0000103, 0.0009265, -0.0005207, 0.0005239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004854, upper bound: 0.0004727
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004854, upper bound: 0.0004727
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010756, -0.0005857, -0.0010767, -0.0005874, -0.0002689, 0.0002678
1: -0.0070399, -0.0057969, -0.0070428, -0.0058013, -0.0006824, 0.0006795
2: 0.0306624, 0.0314336, 0.0306607, 0.0314309, -0.0004233, 0.0004216
3: 0.0008680, 0.0023080, 0.0008731, 0.0023113, -0.0007872, 0.0007905
4: -0.0060538, -0.0047894, -0.0060567, -0.0047939, -0.0006941, 0.0006912
5: 0.0114452, 0.0119241, 0.0114441, 0.0119224, -0.0002629, 0.0002618
6: 0.0014897, 0.0033173, 0.0014962, 0.0033215, -0.0009990, 0.0010032
7: 0.9791017, 0.9803806, 0.9791062, 0.9803835, -0.0006991, 0.0007020
8: -0.0089705, -0.0075994, -0.0089657, -0.0075962, -0.0007495, 0.0007527
9: 0.0000202, 0.0009259, 0.0000181, 0.0009227, -0.0004972, 0.0004951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004796, upper bound: 0.0004794
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004796, upper bound: 0.0004794
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010905, -0.0005773, -0.0010767, -0.0005874, -0.0002915, 0.0002835
1: -0.0070778, -0.0057756, -0.0070428, -0.0058013, -0.0007398, 0.0007195
2: 0.0306389, 0.0314468, 0.0306607, 0.0314309, -0.0004590, 0.0004464
3: 0.0008433, 0.0023519, 0.0008731, 0.0023113, -0.0008335, 0.0008570
4: -0.0060924, -0.0047678, -0.0060567, -0.0047939, -0.0007525, 0.0007318
5: 0.0114306, 0.0119323, 0.0114441, 0.0119224, -0.0002850, 0.0002772
6: 0.0014584, 0.0033730, 0.0014962, 0.0033215, -0.0010578, 0.0010877
7: 0.9790798, 0.9804195, 0.9791062, 0.9803835, -0.0007402, 0.0007611
8: -0.0089940, -0.0075576, -0.0089657, -0.0075962, -0.0007936, 0.0008160
9: -0.0000074, 0.0009414, 0.0000181, 0.0009227, -0.0005390, 0.0005242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004796, upper bound: 0.0004794
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004796, upper bound: 0.0004794
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010756, -0.0005857, -0.0010836, -0.0005857, -0.0002578, 0.0002674
1: -0.0070399, -0.0057969, -0.0070604, -0.0057968, -0.0006542, 0.0006785
2: 0.0306624, 0.0314336, 0.0306497, 0.0314337, -0.0004059, 0.0004209
3: 0.0008680, 0.0023080, 0.0008679, 0.0023317, -0.0007860, 0.0007578
4: -0.0060538, -0.0047894, -0.0060747, -0.0047893, -0.0006654, 0.0006902
5: 0.0114452, 0.0119241, 0.0114373, 0.0119241, -0.0002520, 0.0002614
6: 0.0014897, 0.0033173, 0.0014896, 0.0033474, -0.0009976, 0.0009618
7: 0.9791017, 0.9803806, 0.9791017, 0.9804016, -0.0006980, 0.0006730
8: -0.0089705, -0.0075994, -0.0089706, -0.0075768, -0.0007484, 0.0007216
9: 0.0000202, 0.0009259, 0.0000053, 0.0009260, -0.0004766, 0.0004944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004811, upper bound: 0.0004804
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004811, upper bound: 0.0004804
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010905, -0.0005773, -0.0010836, -0.0005857, -0.0002794, 0.0002797
1: -0.0070778, -0.0057756, -0.0070604, -0.0057968, -0.0007090, 0.0007098
2: 0.0306389, 0.0314468, 0.0306497, 0.0314337, -0.0004399, 0.0004404
3: 0.0008433, 0.0023519, 0.0008679, 0.0023317, -0.0008223, 0.0008213
4: -0.0060924, -0.0047678, -0.0060747, -0.0047893, -0.0007211, 0.0007220
5: 0.0114306, 0.0119323, 0.0114373, 0.0119241, -0.0002732, 0.0002735
6: 0.0014584, 0.0033730, 0.0014896, 0.0033474, -0.0010436, 0.0010424
7: 0.9790798, 0.9804195, 0.9791017, 0.9804016, -0.0007302, 0.0007294
8: -0.0089940, -0.0075576, -0.0089706, -0.0075768, -0.0007829, 0.0007820
9: -0.0000074, 0.0009414, 0.0000053, 0.0009260, -0.0005166, 0.0005172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004811, upper bound: 0.0004804
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004811, upper bound: 0.0004804
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010686, -0.0005882, -0.0010908, -0.0005657, -0.0002791, 0.0002840
1: -0.0070223, -0.0058031, -0.0070786, -0.0057461, -0.0007081, 0.0007208
2: 0.0306734, 0.0314297, 0.0306384, 0.0314651, -0.0004393, 0.0004472
3: 0.0008752, 0.0022875, 0.0008092, 0.0023528, -0.0008350, 0.0008203
4: -0.0060359, -0.0047958, -0.0060932, -0.0047378, -0.0007203, 0.0007332
5: 0.0114520, 0.0119217, 0.0114303, 0.0119436, -0.0002728, 0.0002777
6: 0.0014989, 0.0032913, 0.0014151, 0.0033742, -0.0010597, 0.0010411
7: 0.9791081, 0.9803624, 0.9790494, 0.9804203, -0.0007415, 0.0007285
8: -0.0089636, -0.0076189, -0.0090265, -0.0075567, -0.0007950, 0.0007811
9: 0.0000331, 0.0009214, -0.0000080, 0.0009629, -0.0005160, 0.0005252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004637
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004637
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010834, -0.0005782, -0.0010908, -0.0005657, -0.0003011, 0.0002958
1: -0.0070599, -0.0057779, -0.0070786, -0.0057461, -0.0007641, 0.0007506
2: 0.0306500, 0.0314454, 0.0306384, 0.0314651, -0.0004741, 0.0004657
3: 0.0008460, 0.0023311, 0.0008092, 0.0023528, -0.0008695, 0.0008852
4: -0.0060741, -0.0047701, -0.0060932, -0.0047378, -0.0007773, 0.0007635
5: 0.0114375, 0.0119314, 0.0114303, 0.0119436, -0.0002944, 0.0002892
6: 0.0014618, 0.0033466, 0.0014151, 0.0033742, -0.0011035, 0.0011234
7: 0.9790821, 0.9804011, 0.9790494, 0.9804203, -0.0007722, 0.0007861
8: -0.0089914, -0.0075774, -0.0090265, -0.0075567, -0.0008279, 0.0008429
9: 0.0000057, 0.0009398, -0.0000080, 0.0009629, -0.0005568, 0.0005469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004636
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004970, upper bound: 0.0004637
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010686, -0.0005882, -0.0010967, -0.0005627, -0.0002790, 0.0002882
1: -0.0070223, -0.0058031, -0.0070936, -0.0057386, -0.0007081, 0.0007312
2: 0.0306734, 0.0314297, 0.0306291, 0.0314698, -0.0004393, 0.0004537
3: 0.0008752, 0.0022875, 0.0008004, 0.0023702, -0.0008471, 0.0008203
4: -0.0060359, -0.0047958, -0.0061084, -0.0047301, -0.0007202, 0.0007438
5: 0.0114520, 0.0119217, 0.0114245, 0.0119465, -0.0002728, 0.0002817
6: 0.0014989, 0.0032913, 0.0014040, 0.0033962, -0.0010751, 0.0010410
7: 0.9791081, 0.9803624, 0.9790418, 0.9804357, -0.0007523, 0.0007285
8: -0.0089636, -0.0076189, -0.0090348, -0.0075402, -0.0008066, 0.0007810
9: 0.0000331, 0.0009214, -0.0000189, 0.0009684, -0.0005159, 0.0005328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005032, upper bound: 0.0004636
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005032, upper bound: 0.0004637
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010834, -0.0005782, -0.0010967, -0.0005627, -0.0003011, 0.0002999
1: -0.0070599, -0.0057779, -0.0070936, -0.0057386, -0.0007641, 0.0007610
2: 0.0306500, 0.0314454, 0.0306291, 0.0314698, -0.0004740, 0.0004721
3: 0.0008460, 0.0023311, 0.0008004, 0.0023702, -0.0008816, 0.0008851
4: -0.0060741, -0.0047701, -0.0061084, -0.0047301, -0.0007772, 0.0007741
5: 0.0114375, 0.0119314, 0.0114245, 0.0119465, -0.0002944, 0.0002932
6: 0.0014618, 0.0033466, 0.0014040, 0.0033962, -0.0011189, 0.0011234
7: 0.9790821, 0.9804011, 0.9790418, 0.9804357, -0.0007829, 0.0007861
8: -0.0089914, -0.0075774, -0.0090348, -0.0075402, -0.0008394, 0.0008428
9: 0.0000057, 0.0009398, -0.0000189, 0.0009684, -0.0005567, 0.0005545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005032, upper bound: 0.0004637
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0005032, upper bound: 0.0004637
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010756, -0.0005857, -0.0010897, -0.0005659, -0.0002929, 0.0002834
1: -0.0070399, -0.0057969, -0.0070759, -0.0057465, -0.0007432, 0.0007191
2: 0.0306624, 0.0314336, 0.0306401, 0.0314649, -0.0004611, 0.0004461
3: 0.0008680, 0.0023080, 0.0008096, 0.0023496, -0.0008331, 0.0008610
4: -0.0060538, -0.0047894, -0.0060904, -0.0047382, -0.0007560, 0.0007315
5: 0.0114452, 0.0119241, 0.0114313, 0.0119435, -0.0002863, 0.0002771
6: 0.0014897, 0.0033173, 0.0014156, 0.0033701, -0.0010573, 0.0010927
7: 0.9791017, 0.9803806, 0.9790499, 0.9804175, -0.0007398, 0.0007646
8: -0.0089705, -0.0075994, -0.0090261, -0.0075598, -0.0007932, 0.0008198
9: 0.0000202, 0.0009259, -0.0000059, 0.0009626, -0.0005415, 0.0005240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004937, upper bound: 0.0004655
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004937, upper bound: 0.0004655
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010905, -0.0005773, -0.0010897, -0.0005659, -0.0003155, 0.0002991
1: -0.0070778, -0.0057756, -0.0070759, -0.0057465, -0.0008006, 0.0007591
2: 0.0306389, 0.0314468, 0.0306401, 0.0314649, -0.0004967, 0.0004709
3: 0.0008433, 0.0023519, 0.0008096, 0.0023496, -0.0008794, 0.0009275
4: -0.0060924, -0.0047678, -0.0060904, -0.0047382, -0.0008144, 0.0007721
5: 0.0114306, 0.0119323, 0.0114313, 0.0119435, -0.0003085, 0.0002925
6: 0.0014584, 0.0033730, 0.0014156, 0.0033701, -0.0011160, 0.0011771
7: 0.9790798, 0.9804195, 0.9790499, 0.9804175, -0.0007809, 0.0008237
8: -0.0089940, -0.0075576, -0.0090261, -0.0075598, -0.0008373, 0.0008831
9: -0.0000074, 0.0009414, -0.0000059, 0.0009626, -0.0005834, 0.0005531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004937, upper bound: 0.0004654
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004937, upper bound: 0.0004655
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010756, -0.0005857, -0.0010991, -0.0005630, -0.0002810, 0.0002832
1: -0.0070399, -0.0057969, -0.0070998, -0.0057392, -0.0007132, 0.0007186
2: 0.0306624, 0.0314336, 0.0306253, 0.0314694, -0.0004424, 0.0004458
3: 0.0008680, 0.0023080, 0.0008011, 0.0023774, -0.0008324, 0.0008262
4: -0.0060538, -0.0047894, -0.0061147, -0.0047307, -0.0007254, 0.0007309
5: 0.0114452, 0.0119241, 0.0114221, 0.0119463, -0.0002748, 0.0002768
6: 0.0014897, 0.0033173, 0.0014049, 0.0034053, -0.0010565, 0.0010485
7: 0.9791017, 0.9803806, 0.9790424, 0.9804422, -0.0007393, 0.0007337
8: -0.0089705, -0.0075994, -0.0090341, -0.0075333, -0.0007926, 0.0007866
9: 0.0000202, 0.0009259, -0.0000234, 0.0009680, -0.0005196, 0.0005236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004955, upper bound: 0.0004670
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004955, upper bound: 0.0004670
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010905, -0.0005773, -0.0010991, -0.0005630, -0.0003026, 0.0002955
1: -0.0070778, -0.0057756, -0.0070998, -0.0057392, -0.0007679, 0.0007499
2: 0.0306389, 0.0314468, 0.0306253, 0.0314694, -0.0004764, 0.0004652
3: 0.0008433, 0.0023519, 0.0008011, 0.0023774, -0.0008687, 0.0008896
4: -0.0060924, -0.0047678, -0.0061147, -0.0047307, -0.0007811, 0.0007627
5: 0.0114306, 0.0119323, 0.0114221, 0.0119463, -0.0002959, 0.0002889
6: 0.0014584, 0.0033730, 0.0014049, 0.0034053, -0.0011025, 0.0011290
7: 0.9790798, 0.9804195, 0.9790424, 0.9804422, -0.0007714, 0.0007900
8: -0.0089940, -0.0075576, -0.0090341, -0.0075333, -0.0008271, 0.0008471
9: -0.0000074, 0.0009414, -0.0000234, 0.0009680, -0.0005595, 0.0005464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004955, upper bound: 0.0004670
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004955, upper bound: 0.0004670
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010723, -0.0005875, -0.0010769, -0.0005854, -0.0002637, 0.0002664
1: -0.0070316, -0.0058014, -0.0070434, -0.0057960, -0.0006691, 0.0006761
2: 0.0306676, 0.0314308, 0.0306603, 0.0314341, -0.0004151, 0.0004195
3: 0.0008732, 0.0022984, 0.0008670, 0.0023120, -0.0007833, 0.0007751
4: -0.0060454, -0.0047940, -0.0060573, -0.0047886, -0.0006806, 0.0006878
5: 0.0114484, 0.0119223, 0.0114438, 0.0119244, -0.0002578, 0.0002605
6: 0.0014964, 0.0033050, 0.0014885, 0.0033223, -0.0009941, 0.0009837
7: 0.9791064, 0.9803720, 0.9791008, 0.9803841, -0.0006956, 0.0006883
8: -0.0089655, -0.0076086, -0.0089714, -0.0075956, -0.0007458, 0.0007380
9: 0.0000263, 0.0009226, 0.0000177, 0.0009265, -0.0004875, 0.0004926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004818, upper bound: 0.0004757
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004818, upper bound: 0.0004757
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010866, -0.0005781, -0.0010769, -0.0005854, -0.0002853, 0.0002785
1: -0.0070681, -0.0057777, -0.0070434, -0.0057960, -0.0007240, 0.0007067
2: 0.0306450, 0.0314455, 0.0306603, 0.0314341, -0.0004491, 0.0004384
3: 0.0008457, 0.0023406, 0.0008670, 0.0023120, -0.0008186, 0.0008387
4: -0.0060824, -0.0047699, -0.0060573, -0.0047886, -0.0007364, 0.0007188
5: 0.0114343, 0.0119315, 0.0114438, 0.0119244, -0.0002789, 0.0002723
6: 0.0014615, 0.0033586, 0.0014885, 0.0033223, -0.0010390, 0.0010644
7: 0.9790819, 0.9804095, 0.9791008, 0.9803841, -0.0007270, 0.0007448
8: -0.0089917, -0.0075684, -0.0089714, -0.0075956, -0.0007795, 0.0007986
9: -0.0000003, 0.0009399, 0.0000177, 0.0009265, -0.0005275, 0.0005149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004818, upper bound: 0.0004757
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004818, upper bound: 0.0004757
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010723, -0.0005875, -0.0010809, -0.0005854, -0.0002645, 0.0002728
1: -0.0070316, -0.0058014, -0.0070535, -0.0057961, -0.0006711, 0.0006922
2: 0.0306676, 0.0314308, 0.0306540, 0.0314341, -0.0004164, 0.0004294
3: 0.0008732, 0.0022984, 0.0008671, 0.0023238, -0.0008018, 0.0007774
4: -0.0060454, -0.0047940, -0.0060677, -0.0047886, -0.0006826, 0.0007041
5: 0.0114484, 0.0119223, 0.0114399, 0.0119244, -0.0002586, 0.0002667
6: 0.0014964, 0.0033050, 0.0014886, 0.0033373, -0.0010176, 0.0009867
7: 0.9791064, 0.9803720, 0.9791009, 0.9803945, -0.0007121, 0.0006904
8: -0.0089655, -0.0076086, -0.0089714, -0.0075844, -0.0007635, 0.0007402
9: 0.0000263, 0.0009226, 0.0000103, 0.0009265, -0.0004890, 0.0005043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004821, upper bound: 0.0004758
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004821, upper bound: 0.0004758
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010866, -0.0005781, -0.0010809, -0.0005854, -0.0002864, 0.0002843
1: -0.0070681, -0.0057777, -0.0070535, -0.0057961, -0.0007268, 0.0007214
2: 0.0306450, 0.0314455, 0.0306540, 0.0314341, -0.0004509, 0.0004475
3: 0.0008457, 0.0023406, 0.0008671, 0.0023238, -0.0008357, 0.0008420
4: -0.0060824, -0.0047699, -0.0060677, -0.0047886, -0.0007393, 0.0007337
5: 0.0114343, 0.0119315, 0.0114399, 0.0119244, -0.0002800, 0.0002779
6: 0.0014615, 0.0033586, 0.0014886, 0.0033373, -0.0010606, 0.0010686
7: 0.9790819, 0.9804095, 0.9791009, 0.9803945, -0.0007421, 0.0007477
8: -0.0089917, -0.0075684, -0.0089714, -0.0075844, -0.0007957, 0.0008017
9: -0.0000003, 0.0009399, 0.0000103, 0.0009265, -0.0005296, 0.0005256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004821, upper bound: 0.0004758
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004821, upper bound: 0.0004758
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010796, -0.0005858, -0.0010800, -0.0005853, -0.0002635, 0.0002747
1: -0.0070503, -0.0057971, -0.0070512, -0.0057959, -0.0006686, 0.0006970
2: 0.0306560, 0.0314335, 0.0306554, 0.0314342, -0.0004148, 0.0004325
3: 0.0008682, 0.0023200, 0.0008668, 0.0023210, -0.0008075, 0.0007746
4: -0.0060643, -0.0047896, -0.0060653, -0.0047884, -0.0006801, 0.0007090
5: 0.0114412, 0.0119240, 0.0114408, 0.0119245, -0.0002576, 0.0002686
6: 0.0014900, 0.0033325, 0.0014882, 0.0033338, -0.0010248, 0.0009830
7: 0.9791020, 0.9803912, 0.9791006, 0.9803922, -0.0007171, 0.0006879
8: -0.0089703, -0.0075880, -0.0089716, -0.0075870, -0.0007689, 0.0007375
9: 0.0000127, 0.0009258, 0.0000120, 0.0009267, -0.0004872, 0.0005079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004832
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004843
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010933, -0.0005770, -0.0010800, -0.0005853, -0.0002851, 0.0002898
1: -0.0070849, -0.0057749, -0.0070512, -0.0057959, -0.0007236, 0.0007354
2: 0.0306345, 0.0314473, 0.0306554, 0.0314342, -0.0004489, 0.0004563
3: 0.0008425, 0.0023601, 0.0008668, 0.0023210, -0.0008520, 0.0008382
4: -0.0060996, -0.0047671, -0.0060653, -0.0047884, -0.0007360, 0.0007481
5: 0.0114278, 0.0119325, 0.0114408, 0.0119245, -0.0002788, 0.0002833
6: 0.0014574, 0.0033834, 0.0014882, 0.0033338, -0.0010813, 0.0010638
7: 0.9790791, 0.9804268, 0.9791006, 0.9803922, -0.0007566, 0.0007444
8: -0.0089947, -0.0075498, -0.0089716, -0.0075870, -0.0008112, 0.0007981
9: -0.0000126, 0.0009419, 0.0000120, 0.0009267, -0.0005272, 0.0005358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004832
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004795, upper bound: 0.0004843
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010796, -0.0005858, -0.0010842, -0.0005854, -0.0002643, 0.0002795
1: -0.0070503, -0.0057971, -0.0070618, -0.0057960, -0.0006707, 0.0007092
2: 0.0306560, 0.0314335, 0.0306489, 0.0314341, -0.0004161, 0.0004400
3: 0.0008682, 0.0023200, 0.0008670, 0.0023333, -0.0008216, 0.0007770
4: -0.0060643, -0.0047896, -0.0060760, -0.0047885, -0.0006823, 0.0007214
5: 0.0114412, 0.0119240, 0.0114367, 0.0119244, -0.0002584, 0.0002732
6: 0.0014900, 0.0033325, 0.0014884, 0.0033494, -0.0010427, 0.0009861
7: 0.9791020, 0.9803912, 0.9791008, 0.9804030, -0.0007296, 0.0006901
8: -0.0089703, -0.0075880, -0.0089715, -0.0075753, -0.0007823, 0.0007398
9: 0.0000127, 0.0009258, 0.0000043, 0.0009266, -0.0004887, 0.0005167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004798, upper bound: 0.0004832
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004798, upper bound: 0.0004843
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010933, -0.0005770, -0.0010842, -0.0005854, -0.0002858, 0.0002951
1: -0.0070849, -0.0057749, -0.0070618, -0.0057960, -0.0007252, 0.0007490
2: 0.0306345, 0.0314473, 0.0306489, 0.0314341, -0.0004499, 0.0004647
3: 0.0008425, 0.0023601, 0.0008670, 0.0023333, -0.0008677, 0.0008401
4: -0.0060996, -0.0047671, -0.0060760, -0.0047885, -0.0007377, 0.0007618
5: 0.0114278, 0.0119325, 0.0114367, 0.0119244, -0.0002794, 0.0002886
6: 0.0014574, 0.0033834, 0.0014884, 0.0033494, -0.0011012, 0.0010662
7: 0.9790791, 0.9804268, 0.9791008, 0.9804030, -0.0007705, 0.0007461
8: -0.0089947, -0.0075498, -0.0089715, -0.0075753, -0.0008261, 0.0007999
9: -0.0000126, 0.0009419, 0.0000043, 0.0009266, -0.0005284, 0.0005457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004798, upper bound: 0.0004832
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004798, upper bound: 0.0004843
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010723, -0.0005875, -0.0010908, -0.0005657, -0.0002810, 0.0002807
1: -0.0070316, -0.0058014, -0.0070786, -0.0057461, -0.0007130, 0.0007124
2: 0.0306676, 0.0314308, 0.0306384, 0.0314651, -0.0004424, 0.0004420
3: 0.0008732, 0.0022984, 0.0008092, 0.0023528, -0.0008253, 0.0008260
4: -0.0060454, -0.0047940, -0.0060932, -0.0047378, -0.0007253, 0.0007246
5: 0.0114484, 0.0119223, 0.0114303, 0.0119436, -0.0002747, 0.0002745
6: 0.0014964, 0.0033050, 0.0014151, 0.0033742, -0.0010474, 0.0010483
7: 0.9791064, 0.9803720, 0.9790494, 0.9804203, -0.0007329, 0.0007336
8: -0.0089655, -0.0076086, -0.0090265, -0.0075567, -0.0007858, 0.0007865
9: 0.0000263, 0.0009226, -0.0000080, 0.0009629, -0.0005195, 0.0005191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004961, upper bound: 0.0004628
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004961, upper bound: 0.0004628
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010866, -0.0005781, -0.0010908, -0.0005657, -0.0003026, 0.0002928
1: -0.0070681, -0.0057777, -0.0070786, -0.0057461, -0.0007679, 0.0007429
2: 0.0306450, 0.0314455, 0.0306384, 0.0314651, -0.0004764, 0.0004609
3: 0.0008457, 0.0023406, 0.0008092, 0.0023528, -0.0008606, 0.0008896
4: -0.0060824, -0.0047699, -0.0060932, -0.0047378, -0.0007811, 0.0007557
5: 0.0114343, 0.0119315, 0.0114303, 0.0119436, -0.0002959, 0.0002862
6: 0.0014615, 0.0033586, 0.0014151, 0.0033742, -0.0010923, 0.0011290
7: 0.9790819, 0.9804095, 0.9790494, 0.9804203, -0.0007643, 0.0007900
8: -0.0089917, -0.0075684, -0.0090265, -0.0075567, -0.0008195, 0.0008470
9: -0.0000003, 0.0009399, -0.0000080, 0.0009629, -0.0005595, 0.0005413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004961, upper bound: 0.0004628
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004961, upper bound: 0.0004628
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010723, -0.0005875, -0.0010967, -0.0005627, -0.0002819, 0.0002871
1: -0.0070316, -0.0058014, -0.0070936, -0.0057386, -0.0007154, 0.0007285
2: 0.0306676, 0.0314308, 0.0306291, 0.0314698, -0.0004439, 0.0004519
3: 0.0008732, 0.0022984, 0.0008004, 0.0023702, -0.0008439, 0.0008288
4: -0.0060454, -0.0047940, -0.0061084, -0.0047301, -0.0007277, 0.0007410
5: 0.0114484, 0.0119223, 0.0114245, 0.0119465, -0.0002756, 0.0002807
6: 0.0014964, 0.0033050, 0.0014040, 0.0033962, -0.0010710, 0.0010518
7: 0.9791064, 0.9803720, 0.9790418, 0.9804357, -0.0007494, 0.0007360
8: -0.0089655, -0.0076086, -0.0090348, -0.0075402, -0.0008035, 0.0007891
9: 0.0000263, 0.0009226, -0.0000189, 0.0009684, -0.0005213, 0.0005308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004967, upper bound: 0.0004630
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004967, upper bound: 0.0004630
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010866, -0.0005781, -0.0010967, -0.0005627, -0.0003039, 0.0002986
1: -0.0070681, -0.0057777, -0.0070936, -0.0057386, -0.0007711, 0.0007576
2: 0.0306450, 0.0314455, 0.0306291, 0.0314698, -0.0004784, 0.0004700
3: 0.0008457, 0.0023406, 0.0008004, 0.0023702, -0.0008777, 0.0008933
4: -0.0060824, -0.0047699, -0.0061084, -0.0047301, -0.0007844, 0.0007707
5: 0.0114343, 0.0119315, 0.0114245, 0.0119465, -0.0002971, 0.0002919
6: 0.0014615, 0.0033586, 0.0014040, 0.0033962, -0.0011139, 0.0011338
7: 0.9790819, 0.9804095, 0.9790418, 0.9804357, -0.0007795, 0.0007933
8: -0.0089917, -0.0075684, -0.0090348, -0.0075402, -0.0008357, 0.0008506
9: -0.0000003, 0.0009399, -0.0000189, 0.0009684, -0.0005619, 0.0005520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004967, upper bound: 0.0004630
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004967, upper bound: 0.0004630
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010796, -0.0005858, -0.0010950, -0.0005656, -0.0002814, 0.0002882
1: -0.0070503, -0.0057971, -0.0070892, -0.0057459, -0.0007141, 0.0007315
2: 0.0306560, 0.0314335, 0.0306319, 0.0314653, -0.0004430, 0.0004538
3: 0.0008682, 0.0023200, 0.0008089, 0.0023651, -0.0008474, 0.0008272
4: -0.0060643, -0.0047896, -0.0061039, -0.0047375, -0.0007263, 0.0007440
5: 0.0114412, 0.0119240, 0.0114262, 0.0119437, -0.0002751, 0.0002818
6: 0.0014900, 0.0033325, 0.0014147, 0.0033897, -0.0010754, 0.0010498
7: 0.9791020, 0.9803912, 0.9790492, 0.9804312, -0.0007525, 0.0007346
8: -0.0089703, -0.0075880, -0.0090268, -0.0075451, -0.0008068, 0.0007876
9: 0.0000127, 0.0009258, -0.0000157, 0.0009631, -0.0005203, 0.0005329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004934, upper bound: 0.0004647
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004934, upper bound: 0.0004662
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010933, -0.0005770, -0.0010950, -0.0005656, -0.0003030, 0.0003034
1: -0.0070849, -0.0057749, -0.0070892, -0.0057459, -0.0007690, 0.0007698
2: 0.0306345, 0.0314473, 0.0306319, 0.0314653, -0.0004771, 0.0004776
3: 0.0008425, 0.0023601, 0.0008089, 0.0023651, -0.0008918, 0.0008909
4: -0.0060996, -0.0047671, -0.0061039, -0.0047375, -0.0007822, 0.0007831
5: 0.0114278, 0.0119325, 0.0114262, 0.0119437, -0.0002963, 0.0002966
6: 0.0014574, 0.0033834, 0.0014147, 0.0033897, -0.0011318, 0.0011306
7: 0.9790791, 0.9804268, 0.9790492, 0.9804312, -0.0007920, 0.0007912
8: -0.0089947, -0.0075498, -0.0090268, -0.0075451, -0.0008492, 0.0008482
9: -0.0000126, 0.0009419, -0.0000157, 0.0009631, -0.0005603, 0.0005609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004934, upper bound: 0.0004647
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004934, upper bound: 0.0004662
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010796, -0.0005858, -0.0011005, -0.0005627, -0.0002824, 0.0002931
1: -0.0070503, -0.0057971, -0.0071034, -0.0057384, -0.0007166, 0.0007437
2: 0.0306560, 0.0314335, 0.0306231, 0.0314699, -0.0004446, 0.0004614
3: 0.0008682, 0.0023200, 0.0008002, 0.0023815, -0.0008616, 0.0008301
4: -0.0060643, -0.0047896, -0.0061183, -0.0047299, -0.0007289, 0.0007565
5: 0.0114412, 0.0119240, 0.0114207, 0.0119466, -0.0002761, 0.0002865
6: 0.0014900, 0.0033325, 0.0014037, 0.0034105, -0.0010935, 0.0010535
7: 0.9791020, 0.9803912, 0.9790416, 0.9804457, -0.0007652, 0.0007372
8: -0.0089703, -0.0075880, -0.0090350, -0.0075294, -0.0008204, 0.0007904
9: 0.0000127, 0.0009258, -0.0000260, 0.0009685, -0.0005221, 0.0005419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004938, upper bound: 0.0004649
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004938, upper bound: 0.0004667
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010933, -0.0005770, -0.0011005, -0.0005627, -0.0003038, 0.0003088
1: -0.0070849, -0.0057749, -0.0071034, -0.0057384, -0.0007710, 0.0007835
2: 0.0306345, 0.0314473, 0.0306231, 0.0314699, -0.0004783, 0.0004861
3: 0.0008425, 0.0023601, 0.0008002, 0.0023815, -0.0009077, 0.0008932
4: -0.0060996, -0.0047671, -0.0061183, -0.0047299, -0.0007843, 0.0007970
5: 0.0114278, 0.0119325, 0.0114207, 0.0119466, -0.0002971, 0.0003019
6: 0.0014574, 0.0033834, 0.0014037, 0.0034105, -0.0011519, 0.0011336
7: 0.9790791, 0.9804268, 0.9790416, 0.9804457, -0.0008061, 0.0007932
8: -0.0089947, -0.0075498, -0.0090350, -0.0075294, -0.0008642, 0.0008505
9: -0.0000126, 0.0009419, -0.0000260, 0.0009685, -0.0005618, 0.0005709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004938, upper bound: 0.0004649
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004938, upper bound: 0.0004667
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010686, -0.0005882, -0.0010916, -0.0005768, -0.0002777, 0.0002923
1: -0.0070223, -0.0058031, -0.0070807, -0.0057744, -0.0007046, 0.0007416
2: 0.0306734, 0.0314297, 0.0306371, 0.0314476, -0.0004371, 0.0004601
3: 0.0008752, 0.0022875, 0.0008419, 0.0023552, -0.0008592, 0.0008162
4: -0.0060359, -0.0047958, -0.0060953, -0.0047666, -0.0007167, 0.0007544
5: 0.0114520, 0.0119217, 0.0114295, 0.0119327, -0.0002715, 0.0002857
6: 0.0014989, 0.0032913, 0.0014567, 0.0033772, -0.0010904, 0.0010359
7: 0.9791081, 0.9803624, 0.9790786, 0.9804224, -0.0007630, 0.0007249
8: -0.0089636, -0.0076189, -0.0089953, -0.0075545, -0.0008181, 0.0007772
9: 0.0000331, 0.0009214, -0.0000095, 0.0009423, -0.0005134, 0.0005404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004750, upper bound: 0.0004860
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004750, upper bound: 0.0004861
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010834, -0.0005782, -0.0010916, -0.0005768, -0.0002799, 0.0002865
1: -0.0070599, -0.0057779, -0.0070807, -0.0057744, -0.0007102, 0.0007270
2: 0.0306500, 0.0314454, 0.0306371, 0.0314476, -0.0004406, 0.0004510
3: 0.0008460, 0.0023311, 0.0008419, 0.0023552, -0.0008422, 0.0008228
4: -0.0060741, -0.0047701, -0.0060953, -0.0047666, -0.0007224, 0.0007395
5: 0.0114375, 0.0119314, 0.0114295, 0.0119327, -0.0002736, 0.0002801
6: 0.0014618, 0.0033466, 0.0014567, 0.0033772, -0.0010689, 0.0010442
7: 0.9790821, 0.9804011, 0.9790786, 0.9804224, -0.0007479, 0.0007307
8: -0.0089914, -0.0075774, -0.0089953, -0.0075545, -0.0008019, 0.0007834
9: 0.0000057, 0.0009398, -0.0000095, 0.0009423, -0.0005175, 0.0005297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004750, upper bound: 0.0004728
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004750, upper bound: 0.0004728
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010686, -0.0005882, -0.0010949, -0.0005765, -0.0002749, 0.0002931
1: -0.0070223, -0.0058031, -0.0070890, -0.0057736, -0.0006975, 0.0007437
2: 0.0306734, 0.0314297, 0.0306320, 0.0314480, -0.0004328, 0.0004614
3: 0.0008752, 0.0022875, 0.0008411, 0.0023648, -0.0008616, 0.0008081
4: -0.0060359, -0.0047958, -0.0061037, -0.0047658, -0.0007095, 0.0007565
5: 0.0114520, 0.0119217, 0.0114263, 0.0119330, -0.0002687, 0.0002865
6: 0.0014989, 0.0032913, 0.0014556, 0.0033894, -0.0010935, 0.0010255
7: 0.9791081, 0.9803624, 0.9790778, 0.9804310, -0.0007652, 0.0007176
8: -0.0089636, -0.0076189, -0.0089961, -0.0075453, -0.0008204, 0.0007694
9: 0.0000331, 0.0009214, -0.0000155, 0.0009429, -0.0005082, 0.0005419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004785, upper bound: 0.0004860
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004785, upper bound: 0.0004861
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010834, -0.0005782, -0.0010949, -0.0005765, -0.0002777, 0.0002872
1: -0.0070599, -0.0057779, -0.0070890, -0.0057736, -0.0007046, 0.0007287
2: 0.0306500, 0.0314454, 0.0306320, 0.0314480, -0.0004372, 0.0004521
3: 0.0008460, 0.0023311, 0.0008411, 0.0023648, -0.0008442, 0.0008163
4: -0.0060741, -0.0047701, -0.0061037, -0.0047658, -0.0007167, 0.0007412
5: 0.0114375, 0.0119314, 0.0114263, 0.0119330, -0.0002715, 0.0002808
6: 0.0014618, 0.0033466, 0.0014556, 0.0033894, -0.0010714, 0.0010360
7: 0.9790821, 0.9804011, 0.9790778, 0.9804310, -0.0007497, 0.0007249
8: -0.0089914, -0.0075774, -0.0089961, -0.0075453, -0.0008038, 0.0007772
9: 0.0000057, 0.0009398, -0.0000155, 0.0009429, -0.0005134, 0.0005310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004785, upper bound: 0.0004727
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004785, upper bound: 0.0004727
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010756, -0.0005857, -0.0010913, -0.0005780, -0.0002810, 0.0002899
1: -0.0070399, -0.0057969, -0.0070798, -0.0057774, -0.0007130, 0.0007356
2: 0.0306624, 0.0314336, 0.0306377, 0.0314457, -0.0004423, 0.0004563
3: 0.0008680, 0.0023080, 0.0008454, 0.0023542, -0.0008521, 0.0008259
4: -0.0060538, -0.0047894, -0.0060944, -0.0047696, -0.0007252, 0.0007482
5: 0.0114452, 0.0119241, 0.0114298, 0.0119316, -0.0002747, 0.0002834
6: 0.0014897, 0.0033173, 0.0014610, 0.0033760, -0.0010814, 0.0010482
7: 0.9791017, 0.9803806, 0.9790816, 0.9804216, -0.0007567, 0.0007335
8: -0.0089705, -0.0075994, -0.0089920, -0.0075554, -0.0008113, 0.0007864
9: 0.0000202, 0.0009259, -0.0000089, 0.0009401, -0.0005195, 0.0005359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004728, upper bound: 0.0004932
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004728, upper bound: 0.0004932
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010905, -0.0005773, -0.0010913, -0.0005780, -0.0002853, 0.0002857
1: -0.0070778, -0.0057756, -0.0070798, -0.0057774, -0.0007240, 0.0007251
2: 0.0306389, 0.0314468, 0.0306377, 0.0314457, -0.0004492, 0.0004498
3: 0.0008433, 0.0023519, 0.0008454, 0.0023542, -0.0008399, 0.0008388
4: -0.0060924, -0.0047678, -0.0060944, -0.0047696, -0.0007365, 0.0007375
5: 0.0114306, 0.0119323, 0.0114298, 0.0119316, -0.0002790, 0.0002793
6: 0.0014584, 0.0033730, 0.0014610, 0.0033760, -0.0010660, 0.0010645
7: 0.9790798, 0.9804195, 0.9790816, 0.9804216, -0.0007459, 0.0007449
8: -0.0089940, -0.0075576, -0.0089920, -0.0075554, -0.0007998, 0.0007986
9: -0.0000074, 0.0009414, -0.0000089, 0.0009401, -0.0005275, 0.0005283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004728, upper bound: 0.0004794
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004728, upper bound: 0.0004794
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010756, -0.0005857, -0.0010981, -0.0005769, -0.0002705, 0.0002892
1: -0.0070399, -0.0057969, -0.0070972, -0.0057745, -0.0006865, 0.0007338
2: 0.0306624, 0.0314336, 0.0306269, 0.0314475, -0.0004259, 0.0004553
3: 0.0008680, 0.0023080, 0.0008421, 0.0023744, -0.0008501, 0.0007952
4: -0.0060538, -0.0047894, -0.0061121, -0.0047667, -0.0006983, 0.0007464
5: 0.0114452, 0.0119241, 0.0114231, 0.0119327, -0.0002645, 0.0002827
6: 0.0014897, 0.0033173, 0.0014568, 0.0034015, -0.0010789, 0.0010093
7: 0.9791017, 0.9803806, 0.9790788, 0.9804395, -0.0007549, 0.0007062
8: -0.0089705, -0.0075994, -0.0089952, -0.0075362, -0.0008094, 0.0007572
9: 0.0000202, 0.0009259, -0.0000215, 0.0009422, -0.0005002, 0.0005347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004744, upper bound: 0.0004945
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004744, upper bound: 0.0004945
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010905, -0.0005773, -0.0010981, -0.0005769, -0.0002743, 0.0002844
1: -0.0070778, -0.0057756, -0.0070972, -0.0057745, -0.0006960, 0.0007217
2: 0.0306389, 0.0314468, 0.0306269, 0.0314475, -0.0004318, 0.0004477
3: 0.0008433, 0.0023519, 0.0008421, 0.0023744, -0.0008360, 0.0008063
4: -0.0060924, -0.0047678, -0.0061121, -0.0047667, -0.0007079, 0.0007340
5: 0.0114306, 0.0119323, 0.0114231, 0.0119327, -0.0002681, 0.0002780
6: 0.0014584, 0.0033730, 0.0014568, 0.0034015, -0.0010610, 0.0010232
7: 0.9790798, 0.9804195, 0.9790788, 0.9804395, -0.0007424, 0.0007160
8: -0.0089940, -0.0075576, -0.0089952, -0.0075362, -0.0007960, 0.0007677
9: -0.0000074, 0.0009414, -0.0000215, 0.0009422, -0.0005071, 0.0005258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004744, upper bound: 0.0004804
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004744, upper bound: 0.0004804
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010686, -0.0005882, -0.0011049, -0.0005571, -0.0002896, 0.0003011
1: -0.0070223, -0.0058031, -0.0071144, -0.0057244, -0.0007348, 0.0007640
2: 0.0306734, 0.0314297, 0.0306162, 0.0314786, -0.0004559, 0.0004740
3: 0.0008752, 0.0022875, 0.0007840, 0.0023943, -0.0008851, 0.0008513
4: -0.0060359, -0.0047958, -0.0061296, -0.0047157, -0.0007474, 0.0007771
5: 0.0114520, 0.0119217, 0.0114165, 0.0119520, -0.0002831, 0.0002944
6: 0.0014989, 0.0032913, 0.0013831, 0.0034268, -0.0011233, 0.0010803
7: 0.9791081, 0.9803624, 0.9790272, 0.9804571, -0.0007860, 0.0007560
8: -0.0089636, -0.0076189, -0.0090505, -0.0075172, -0.0008427, 0.0008105
9: 0.0000331, 0.0009214, -0.0000341, 0.0009788, -0.0005354, 0.0005567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004904, upper bound: 0.0004721
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004904, upper bound: 0.0004721
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010834, -0.0005782, -0.0011049, -0.0005571, -0.0002954, 0.0002981
1: -0.0070599, -0.0057779, -0.0071144, -0.0057244, -0.0007495, 0.0007566
2: 0.0306500, 0.0314454, 0.0306162, 0.0314786, -0.0004650, 0.0004694
3: 0.0008460, 0.0023311, 0.0007840, 0.0023943, -0.0008764, 0.0008683
4: -0.0060741, -0.0047701, -0.0061296, -0.0047157, -0.0007624, 0.0007695
5: 0.0114375, 0.0119314, 0.0114165, 0.0119520, -0.0002888, 0.0002915
6: 0.0014618, 0.0033466, 0.0013831, 0.0034268, -0.0011123, 0.0011020
7: 0.9790821, 0.9804011, 0.9790272, 0.9804571, -0.0007783, 0.0007711
8: -0.0089914, -0.0075774, -0.0090505, -0.0075172, -0.0008345, 0.0008267
9: 0.0000057, 0.0009398, -0.0000341, 0.0009788, -0.0005461, 0.0005512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004904, upper bound: 0.0004637
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004904, upper bound: 0.0004637
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010686, -0.0005882, -0.0011088, -0.0005553, -0.0002910, 0.0003061
1: -0.0070223, -0.0058031, -0.0071243, -0.0057197, -0.0007385, 0.0007769
2: 0.0306734, 0.0314297, 0.0306101, 0.0314815, -0.0004582, 0.0004820
3: 0.0008752, 0.0022875, 0.0007786, 0.0024057, -0.0008999, 0.0008555
4: -0.0060359, -0.0047958, -0.0061396, -0.0047109, -0.0007512, 0.0007902
5: 0.0114520, 0.0119217, 0.0114127, 0.0119538, -0.0002845, 0.0002993
6: 0.0014989, 0.0032913, 0.0013762, 0.0034413, -0.0011421, 0.0010857
7: 0.9791081, 0.9803624, 0.9790223, 0.9804674, -0.0007992, 0.0007598
8: -0.0089636, -0.0076189, -0.0090557, -0.0075063, -0.0008569, 0.0008146
9: 0.0000331, 0.0009214, -0.0000412, 0.0009822, -0.0005381, 0.0005660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004953, upper bound: 0.0004721
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004953, upper bound: 0.0004720
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010834, -0.0005782, -0.0011088, -0.0005553, -0.0002963, 0.0003027
1: -0.0070599, -0.0057779, -0.0071243, -0.0057197, -0.0007518, 0.0007681
2: 0.0306500, 0.0314454, 0.0306101, 0.0314815, -0.0004664, 0.0004765
3: 0.0008460, 0.0023311, 0.0007786, 0.0024057, -0.0008898, 0.0008709
4: -0.0060741, -0.0047701, -0.0061396, -0.0047109, -0.0007647, 0.0007813
5: 0.0114375, 0.0119314, 0.0114127, 0.0119538, -0.0002897, 0.0002959
6: 0.0014618, 0.0033466, 0.0013762, 0.0034413, -0.0011293, 0.0011053
7: 0.9790821, 0.9804011, 0.9790223, 0.9804674, -0.0007902, 0.0007735
8: -0.0089914, -0.0075774, -0.0090557, -0.0075063, -0.0008472, 0.0008293
9: 0.0000057, 0.0009398, -0.0000412, 0.0009822, -0.0005478, 0.0005596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004953, upper bound: 0.0004637
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004953, upper bound: 0.0004637
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010756, -0.0005857, -0.0011033, -0.0005572, -0.0003013, 0.0003003
1: -0.0070399, -0.0057969, -0.0071104, -0.0057244, -0.0007646, 0.0007622
2: 0.0306624, 0.0314336, 0.0306187, 0.0314786, -0.0004744, 0.0004728
3: 0.0008680, 0.0023080, 0.0007841, 0.0023896, -0.0008829, 0.0008858
4: -0.0060538, -0.0047894, -0.0061255, -0.0047157, -0.0007778, 0.0007752
5: 0.0114452, 0.0119241, 0.0114180, 0.0119520, -0.0002946, 0.0002936
6: 0.0014897, 0.0033173, 0.0013832, 0.0034209, -0.0011205, 0.0011242
7: 0.9791017, 0.9803806, 0.9790272, 0.9804531, -0.0007841, 0.0007866
8: -0.0089705, -0.0075994, -0.0090504, -0.0075217, -0.0008407, 0.0008434
9: 0.0000202, 0.0009259, -0.0000311, 0.0009787, -0.0005571, 0.0005553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004755
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004755
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010905, -0.0005773, -0.0011033, -0.0005572, -0.0003090, 0.0002991
1: -0.0070778, -0.0057756, -0.0071104, -0.0057244, -0.0007841, 0.0007590
2: 0.0306389, 0.0314468, 0.0306187, 0.0314786, -0.0004864, 0.0004709
3: 0.0008433, 0.0023519, 0.0007841, 0.0023896, -0.0008792, 0.0009083
4: -0.0060924, -0.0047678, -0.0061255, -0.0047157, -0.0007975, 0.0007720
5: 0.0114306, 0.0119323, 0.0114180, 0.0119520, -0.0003021, 0.0002924
6: 0.0014584, 0.0033730, 0.0013832, 0.0034209, -0.0011158, 0.0011528
7: 0.9790798, 0.9804195, 0.9790272, 0.9804531, -0.0007808, 0.0008067
8: -0.0089940, -0.0075576, -0.0090504, -0.0075217, -0.0008372, 0.0008649
9: -0.0000074, 0.0009414, -0.0000311, 0.0009787, -0.0005713, 0.0005530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004655
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004876, upper bound: 0.0004655
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010756, -0.0005857, -0.0011131, -0.0005556, -0.0002898, 0.0003002
1: -0.0070399, -0.0057969, -0.0071353, -0.0057205, -0.0007355, 0.0007618
2: 0.0306624, 0.0314336, 0.0306033, 0.0314810, -0.0004563, 0.0004726
3: 0.0008680, 0.0023080, 0.0007795, 0.0024185, -0.0008825, 0.0008521
4: -0.0060538, -0.0047894, -0.0061508, -0.0047117, -0.0007482, 0.0007749
5: 0.0114452, 0.0119241, 0.0114084, 0.0119535, -0.0002834, 0.0002935
6: 0.0014897, 0.0033173, 0.0013774, 0.0034575, -0.0011200, 0.0010814
7: 0.9791017, 0.9803806, 0.9790231, 0.9804786, -0.0007837, 0.0007567
8: -0.0089705, -0.0075994, -0.0090548, -0.0074942, -0.0008403, 0.0008113
9: 0.0000202, 0.0009259, -0.0000492, 0.0009816, -0.0005359, 0.0005550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004892, upper bound: 0.0004769
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004892, upper bound: 0.0004769
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010905, -0.0005773, -0.0011131, -0.0005556, -0.0002969, 0.0002985
1: -0.0070778, -0.0057756, -0.0071353, -0.0057205, -0.0007535, 0.0007574
2: 0.0306389, 0.0314468, 0.0306033, 0.0314810, -0.0004675, 0.0004699
3: 0.0008433, 0.0023519, 0.0007795, 0.0024185, -0.0008774, 0.0008729
4: -0.0060924, -0.0047678, -0.0061508, -0.0047117, -0.0007665, 0.0007704
5: 0.0114306, 0.0119323, 0.0114084, 0.0119535, -0.0002903, 0.0002918
6: 0.0014584, 0.0033730, 0.0013774, 0.0034575, -0.0011135, 0.0011079
7: 0.9790798, 0.9804195, 0.9790231, 0.9804786, -0.0007792, 0.0007752
8: -0.0089940, -0.0075576, -0.0090548, -0.0074942, -0.0008354, 0.0008312
9: -0.0000074, 0.0009414, -0.0000492, 0.0009816, -0.0005490, 0.0005518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004892, upper bound: 0.0004670
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004892, upper bound: 0.0004670
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010723, -0.0005875, -0.0010916, -0.0005768, -0.0002796, 0.0002890
1: -0.0070316, -0.0058014, -0.0070807, -0.0057744, -0.0007095, 0.0007333
2: 0.0306676, 0.0314308, 0.0306371, 0.0314476, -0.0004402, 0.0004549
3: 0.0008732, 0.0022984, 0.0008419, 0.0023552, -0.0008494, 0.0008219
4: -0.0060454, -0.0047940, -0.0060953, -0.0047666, -0.0007217, 0.0007458
5: 0.0114484, 0.0119223, 0.0114295, 0.0119327, -0.0002734, 0.0002825
6: 0.0014964, 0.0033050, 0.0014567, 0.0033772, -0.0010780, 0.0010431
7: 0.9791064, 0.9803720, 0.9790786, 0.9804224, -0.0007544, 0.0007299
8: -0.0089655, -0.0076086, -0.0089953, -0.0075545, -0.0008088, 0.0007826
9: 0.0000263, 0.0009226, -0.0000095, 0.0009423, -0.0005169, 0.0005343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004748, upper bound: 0.0004886
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004748, upper bound: 0.0004886
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010866, -0.0005781, -0.0010916, -0.0005768, -0.0002810, 0.0002827
1: -0.0070681, -0.0057777, -0.0070807, -0.0057744, -0.0007131, 0.0007173
2: 0.0306450, 0.0314455, 0.0306371, 0.0314476, -0.0004424, 0.0004450
3: 0.0008457, 0.0023406, 0.0008419, 0.0023552, -0.0008310, 0.0008261
4: -0.0060824, -0.0047699, -0.0060953, -0.0047666, -0.0007253, 0.0007296
5: 0.0114343, 0.0119315, 0.0114295, 0.0119327, -0.0002747, 0.0002764
6: 0.0014615, 0.0033586, 0.0014567, 0.0033772, -0.0010546, 0.0010484
7: 0.9790819, 0.9804095, 0.9790786, 0.9804224, -0.0007380, 0.0007336
8: -0.0089917, -0.0075684, -0.0089953, -0.0075545, -0.0007912, 0.0007865
9: -0.0000003, 0.0009399, -0.0000095, 0.0009423, -0.0005195, 0.0005226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004748, upper bound: 0.0004757
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004748, upper bound: 0.0004757
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010723, -0.0005875, -0.0010949, -0.0005765, -0.0002803, 0.0002952
1: -0.0070316, -0.0058014, -0.0070890, -0.0057736, -0.0007113, 0.0007490
2: 0.0306676, 0.0314308, 0.0306320, 0.0314480, -0.0004413, 0.0004647
3: 0.0008732, 0.0022984, 0.0008411, 0.0023648, -0.0008677, 0.0008240
4: -0.0060454, -0.0047940, -0.0061037, -0.0047658, -0.0007235, 0.0007619
5: 0.0114484, 0.0119223, 0.0114263, 0.0119330, -0.0002740, 0.0002886
6: 0.0014964, 0.0033050, 0.0014556, 0.0033894, -0.0011012, 0.0010457
7: 0.9791064, 0.9803720, 0.9790778, 0.9804310, -0.0007706, 0.0007317
8: -0.0089655, -0.0076086, -0.0089961, -0.0075453, -0.0008262, 0.0007845
9: 0.0000263, 0.0009226, -0.0000155, 0.0009429, -0.0005182, 0.0005457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0004886
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0004886
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010866, -0.0005781, -0.0010949, -0.0005765, -0.0002823, 0.0002890
1: -0.0070681, -0.0057777, -0.0070890, -0.0057736, -0.0007163, 0.0007334
2: 0.0306450, 0.0314455, 0.0306320, 0.0314480, -0.0004444, 0.0004550
3: 0.0008457, 0.0023406, 0.0008411, 0.0023648, -0.0008497, 0.0008298
4: -0.0060824, -0.0047699, -0.0061037, -0.0047658, -0.0007286, 0.0007460
5: 0.0114343, 0.0119315, 0.0114263, 0.0119330, -0.0002760, 0.0002826
6: 0.0014615, 0.0033586, 0.0014556, 0.0033894, -0.0010783, 0.0010531
7: 0.9790819, 0.9804095, 0.9790778, 0.9804310, -0.0007546, 0.0007369
8: -0.0089917, -0.0075684, -0.0089961, -0.0075453, -0.0008090, 0.0007901
9: -0.0000003, 0.0009399, -0.0000155, 0.0009429, -0.0005219, 0.0005344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0004758
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004751, upper bound: 0.0004758
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010796, -0.0005858, -0.0010924, -0.0005780, -0.0002867, 0.0002874
1: -0.0070503, -0.0057971, -0.0070827, -0.0057772, -0.0007276, 0.0007294
2: 0.0306560, 0.0314335, 0.0306359, 0.0314458, -0.0004514, 0.0004526
3: 0.0008682, 0.0023200, 0.0008452, 0.0023575, -0.0008450, 0.0008429
4: -0.0060643, -0.0047896, -0.0060973, -0.0047695, -0.0007401, 0.0007420
5: 0.0114412, 0.0119240, 0.0114287, 0.0119316, -0.0002803, 0.0002810
6: 0.0014900, 0.0033325, 0.0014608, 0.0033801, -0.0010724, 0.0010697
7: 0.9791020, 0.9803912, 0.9790815, 0.9804245, -0.0007504, 0.0007485
8: -0.0089703, -0.0075880, -0.0089922, -0.0075523, -0.0008046, 0.0008026
9: 0.0000127, 0.0009258, -0.0000109, 0.0009402, -0.0005301, 0.0005315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0004958
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0004957
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010933, -0.0005770, -0.0010924, -0.0005780, -0.0002917, 0.0002833
1: -0.0070849, -0.0057749, -0.0070827, -0.0057772, -0.0007402, 0.0007190
2: 0.0306345, 0.0314473, 0.0306359, 0.0314458, -0.0004592, 0.0004461
3: 0.0008425, 0.0023601, 0.0008452, 0.0023575, -0.0008329, 0.0008575
4: -0.0060996, -0.0047671, -0.0060973, -0.0047695, -0.0007529, 0.0007314
5: 0.0114278, 0.0119325, 0.0114287, 0.0119316, -0.0002852, 0.0002770
6: 0.0014574, 0.0033834, 0.0014608, 0.0033801, -0.0010571, 0.0010882
7: 0.9790791, 0.9804268, 0.9790815, 0.9804245, -0.0007397, 0.0007615
8: -0.0089947, -0.0075498, -0.0089922, -0.0075523, -0.0007931, 0.0008164
9: -0.0000126, 0.0009419, -0.0000109, 0.0009402, -0.0005393, 0.0005239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0004832
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004727, upper bound: 0.0004832
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010796, -0.0005858, -0.0010990, -0.0005769, -0.0002764, 0.0002859
1: -0.0070503, -0.0057971, -0.0070993, -0.0057745, -0.0007014, 0.0007256
2: 0.0306560, 0.0314335, 0.0306256, 0.0314475, -0.0004351, 0.0004502
3: 0.0008682, 0.0023200, 0.0008420, 0.0023768, -0.0008406, 0.0008125
4: -0.0060643, -0.0047896, -0.0061142, -0.0047666, -0.0007134, 0.0007381
5: 0.0114412, 0.0119240, 0.0114223, 0.0119327, -0.0002702, 0.0002796
6: 0.0014900, 0.0033325, 0.0014568, 0.0034046, -0.0010668, 0.0010312
7: 0.9791020, 0.9803912, 0.9790787, 0.9804416, -0.0007465, 0.0007216
8: -0.0089703, -0.0075880, -0.0089952, -0.0075339, -0.0008004, 0.0007736
9: 0.0000127, 0.0009258, -0.0000231, 0.0009423, -0.0005110, 0.0005287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004743, upper bound: 0.0004969
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004743, upper bound: 0.0004969
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010933, -0.0005770, -0.0010990, -0.0005769, -0.0002806, 0.0002800
1: -0.0070849, -0.0057749, -0.0070993, -0.0057745, -0.0007120, 0.0007104
2: 0.0306345, 0.0314473, 0.0306256, 0.0314475, -0.0004418, 0.0004408
3: 0.0008425, 0.0023601, 0.0008420, 0.0023768, -0.0008230, 0.0008249
4: -0.0060996, -0.0047671, -0.0061142, -0.0047666, -0.0007243, 0.0007226
5: 0.0114278, 0.0119325, 0.0114223, 0.0119327, -0.0002743, 0.0002737
6: 0.0014574, 0.0033834, 0.0014568, 0.0034046, -0.0010445, 0.0010469
7: 0.9790791, 0.9804268, 0.9790787, 0.9804416, -0.0007309, 0.0007325
8: -0.0089947, -0.0075498, -0.0089952, -0.0075339, -0.0007836, 0.0007854
9: -0.0000126, 0.0009419, -0.0000231, 0.0009423, -0.0005188, 0.0005176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004743, upper bound: 0.0004842
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004743, upper bound: 0.0004843
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0010723, -0.0005875, -0.0011049, -0.0005571, -0.0002915, 0.0002978
1: -0.0070316, -0.0058014, -0.0071144, -0.0057244, -0.0007397, 0.0007556
2: 0.0306676, 0.0314308, 0.0306162, 0.0314786, -0.0004589, 0.0004688
3: 0.0008732, 0.0022984, 0.0007840, 0.0023943, -0.0008754, 0.0008569
4: -0.0060454, -0.0047940, -0.0061296, -0.0047157, -0.0007524, 0.0007686
5: 0.0114484, 0.0119223, 0.0114165, 0.0119520, -0.0002850, 0.0002911
6: 0.0014964, 0.0033050, 0.0013831, 0.0034268, -0.0011110, 0.0010876
7: 0.9791064, 0.9803720, 0.9790272, 0.9804571, -0.0007774, 0.0007610
8: -0.0089655, -0.0076086, -0.0090505, -0.0075172, -0.0008335, 0.0008159
9: 0.0000263, 0.0009226, -0.0000341, 0.0009788, -0.0005390, 0.0005506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004901, upper bound: 0.0004722
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0004901, upper bound: 0.0004722
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010866, -0.0005781, -0.0011049, -0.0005571, -0.0002965, 0.0002943
1: -0.0070681, -0.0057777, -0.0071144, -0.0057244, -0.0007524, 0.0007469
2: 0.0306450, 0.0314455, 0.0306162, 0.0314786, -0.0004668, 0.0004634
3: 0.0008457, 0.0023406, 0.0007840, 0.0023943, -0.0008652, 0.0008716
4: -0.0060824, -0.0047699, -0.0061296, -0.0047157, -0.0007653, 0.0007597
5: 0.0114343, 0.0119315, 0.0114165, 0.0119520, -0.0002899, 0.0002877
6: 0.0014615, 0.0033586, 0.0013831, 0.0034268, -0.0010981, 0.0011061
7: 0.9790819, 0.9804095, 0.9790272, 0.9804571, -0.0007684, 0.0007740
8: -0.0089917, -0.0075684, -0.0090505, -0.0075172, -0.0008238, 0.0008299
9: -0.0000003, 0.0009399, -0.0000341, 0.0009788, -0.0005482, 0.0005442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 2.01 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.51 + 597.16 = 600.67 seconds
