## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00217782


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0026602, -0.0011764, -0.0026602, -0.0011764, -0.0008953, 0.0008953)
1: (-0.0030782, 0.0009465, -0.0030782, 0.0009465, -0.0024174, 0.0024174)
2: (0.0040394, 0.0074440, 0.0040394, 0.0074440, -0.0020132, 0.0020132)
3: (-0.0042577, -0.0038740, -0.0042577, -0.0038740, -0.0002424, 0.0002424)
4: (0.0036706, 0.0063922, 0.0036706, 0.0063922, -0.0014995, 0.0014995)
5: (-0.0015942, 0.0012017, -0.0015942, 0.0012017, -0.0015806, 0.0015806)
6: (-0.0059026, -0.0044148, -0.0059026, -0.0044148, -0.0007837, 0.0007837)
7: (-0.0000888, 0.0026014, -0.0000888, 0.0026014, -0.0014885, 0.0014885)
8: (-0.0005164, -0.0001644, -0.0005164, -0.0001644, -0.0002288, 0.0002288)
9: (1.0024346, 1.0094026, 1.0024346, 1.0094026, -0.0040757, 0.0040757)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.36 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0024063, upper bound: 0.0024063

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0022790, upper bound: 0.0020881
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0022981, upper bound: 0.0022981
time: 0.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.17 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 9, lower bound: -0.0022790, upper bound: 0.0020881
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 9, lower bound: -0.0022981, upper bound: 0.0022981

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0026729, -0.0012697, -0.0026590, -0.0012048, -0.0008815, 0.0008047
1: -0.0027771, 0.0010019, -0.0029842, 0.0009444, -0.0021141, 0.0023593
2: 0.0043321, 0.0074806, 0.0041279, 0.0074409, -0.0017158, 0.0019323
3: -0.0042612, -0.0039008, -0.0042571, -0.0038823, -0.0002343, 0.0002146
4: 0.0036351, 0.0061438, 0.0036746, 0.0063173, -0.0013824, 0.0012191
5: -0.0013134, 0.0012393, -0.0015072, 0.0011986, -0.0012713, 0.0014623
6: -0.0057812, -0.0043953, -0.0058651, -0.0044176, -0.0006394, 0.0007148
7: -0.0001093, 0.0024061, -0.0000835, 0.0025413, -0.0014047, 0.0012759
8: -0.0004752, -0.0001591, -0.0005034, -0.0001649, -0.0001815, 0.0002096
9: 1.0023537, 1.0087385, 1.0024414, 1.0091977, -0.0038611, 0.0033841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0020881, upper bound: 0.0020881
time: 0.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0020881, upper bound: 0.0020881
time: 0.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0026588, -0.0011929, -0.0026602, -0.0011764, -0.0008938, 0.0008445
1: -0.0030310, 0.0009438, -0.0030782, 0.0009465, -0.0022084, 0.0024142
2: 0.0040829, 0.0074406, 0.0040394, 0.0074440, -0.0018054, 0.0020050
3: -0.0042570, -0.0038794, -0.0042577, -0.0038740, -0.0002418, 0.0002275
4: 0.0036745, 0.0063515, 0.0036706, 0.0063922, -0.0014894, 0.0012633
5: -0.0015524, 0.0011983, -0.0015942, 0.0012017, -0.0013237, 0.0015735
6: -0.0058811, -0.0044172, -0.0059026, -0.0044148, -0.0006606, 0.0007767
7: -0.0000843, 0.0025683, -0.0000888, 0.0026014, -0.0014748, 0.0013342
8: -0.0005096, -0.0001651, -0.0005164, -0.0001644, -0.0001908, 0.0002282
9: 1.0024421, 1.0093085, 1.0024346, 1.0094026, -0.0040591, 0.0035693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0020881, upper bound: 0.0022790
time: 0.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0020881, upper bound: 0.0022981
time: 0.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.47 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.47
Output dim: 9, lower bound: -0.0020881, upper bound: 0.0020881
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.47
Output dim: 9, lower bound: -0.0020881, upper bound: 0.0020881
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 9, lower bound: -0.0020881, upper bound: 0.0022790
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 9, lower bound: -0.0020881, upper bound: 0.0022981

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0026588, -0.0011929, -0.0026729, -0.0012697, -0.0008046, 0.0008982
1: -0.0030310, 0.0009438, -0.0027771, 0.0010019, -0.0024260, 0.0021134
2: 0.0040829, 0.0074406, 0.0043321, 0.0074806, -0.0020023, 0.0017120
3: -0.0042570, -0.0038794, -0.0042612, -0.0039008, -0.0002146, 0.0002389
4: 0.0036745, 0.0063515, 0.0036351, 0.0061438, -0.0012139, 0.0014590
5: -0.0015524, 0.0011983, -0.0013134, 0.0012393, -0.0015444, 0.0012676
6: -0.0058811, -0.0044172, -0.0057812, -0.0043953, -0.0007561, 0.0006359
7: -0.0000843, 0.0025683, -0.0001093, 0.0024061, -0.0012694, 0.0014586
8: -0.0005096, -0.0001651, -0.0004752, -0.0001591, -0.0002224, 0.0001814
9: 1.0024421, 1.0093085, 1.0023537, 1.0087385, -0.0033757, 0.0040313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0017056, upper bound: 0.0016042
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0020112, upper bound: 0.0022008
time: 0.53 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026588, -0.0011929, -0.0026588, -0.0011929, -0.0008429, 0.0008429
1: -0.0030310, 0.0009438, -0.0030310, 0.0009438, -0.0022051, 0.0022051
2: 0.0040829, 0.0074406, 0.0040829, 0.0074406, -0.0017989, 0.0017989
3: -0.0042570, -0.0038794, -0.0042570, -0.0038794, -0.0002268, 0.0002268
4: 0.0036745, 0.0063515, 0.0036745, 0.0063515, -0.0012555, 0.0012555
5: -0.0015524, 0.0011983, -0.0015524, 0.0011983, -0.0013181, 0.0013181
6: -0.0058811, -0.0044172, -0.0058811, -0.0044172, -0.0006551, 0.0006551
7: -0.0000843, 0.0025683, -0.0000843, 0.0025683, -0.0013237, 0.0013237
8: -0.0005096, -0.0001651, -0.0005096, -0.0001651, -0.0001903, 0.0001903
9: 1.0024421, 1.0093085, 1.0024421, 1.0093085, -0.0035562, 0.0035562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013078, upper bound: 0.0021534
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0020112, upper bound: 0.0022152
time: 0.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.44 seconds
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 9, lower bound: -0.0017056, upper bound: 0.0016042
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 9, lower bound: -0.0020112, upper bound: 0.0022008
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.44
Output dim: 9, lower bound: -0.0013078, upper bound: 0.0021534
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.44
Output dim: 9, lower bound: -0.0020112, upper bound: 0.0022152

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0026588, -0.0011929, -0.0026635, -0.0012699, -0.0008044, 0.0006526
1: -0.0030310, 0.0009438, -0.0027771, 0.0009707, -0.0016435, 0.0021133
2: 0.0040829, 0.0074406, 0.0043322, 0.0074550, -0.0014568, 0.0017120
3: -0.0042570, -0.0038794, -0.0042592, -0.0039009, -0.0002144, 0.0002134
4: 0.0036745, 0.0063515, 0.0036476, 0.0061435, -0.0012134, 0.0012487
5: -0.0015524, 0.0011983, -0.0013132, 0.0012240, -0.0012559, 0.0012674
6: -0.0058811, -0.0044172, -0.0057809, -0.0044009, -0.0006921, 0.0006354
7: -0.0000843, 0.0025683, -0.0000955, 0.0024057, -0.0012685, 0.0012431
8: -0.0005096, -0.0001651, -0.0004751, -0.0001611, -0.0001970, 0.0001813
9: 1.0024421, 1.0093085, 1.0023994, 1.0087383, -0.0033755, 0.0030468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0013078, upper bound: 0.0020465
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0013078, upper bound: 0.0022008
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026498, -0.0011931, -0.0026588, -0.0011929, -0.0005650, 0.0008427
1: -0.0030310, 0.0009132, -0.0030310, 0.0009438, -0.0022051, 0.0012382
2: 0.0040830, 0.0074148, 0.0040829, 0.0074406, -0.0017989, 0.0011215
3: -0.0042551, -0.0038796, -0.0042570, -0.0038794, -0.0001994, 0.0002267
4: 0.0036870, 0.0063511, 0.0036745, 0.0063515, -0.0009914, 0.0012551
5: -0.0015522, 0.0011831, -0.0015524, 0.0011983, -0.0013179, 0.0009378
6: -0.0058807, -0.0044221, -0.0058811, -0.0044172, -0.0006547, 0.0005789
7: -0.0000717, 0.0025678, -0.0000843, 0.0025683, -0.0010824, 0.0013229
8: -0.0005094, -0.0001671, -0.0005096, -0.0001651, -0.0001902, 0.0001594
9: 1.0024908, 1.0093082, 1.0024421, 1.0093085, -0.0022900, 0.0035559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0019822, upper bound: 0.0019440
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0019822, upper bound: 0.0022152
time: 0.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.48 seconds
IS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.48
Output dim: 9, lower bound: -0.0013078, upper bound: 0.0020465
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 9, lower bound: -0.0013078, upper bound: 0.0022008
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.48
Output dim: 9, lower bound: -0.0019822, upper bound: 0.0019440
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.48
Output dim: 9, lower bound: -0.0019822, upper bound: 0.0022152

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026498, -0.0011931, -0.0026635, -0.0012699, -0.0005539, 0.0006525
1: -0.0030310, 0.0009132, -0.0027771, 0.0009707, -0.0016435, 0.0013079
2: 0.0040830, 0.0074148, 0.0043322, 0.0074550, -0.0014567, 0.0011518
3: -0.0042551, -0.0038796, -0.0042592, -0.0039009, -0.0001882, 0.0002132
4: 0.0036870, 0.0063511, 0.0036476, 0.0061435, -0.0009945, 0.0012484
5: -0.0015522, 0.0011831, -0.0013132, 0.0012240, -0.0012557, 0.0009747
6: -0.0058807, -0.0044221, -0.0057809, -0.0044009, -0.0006918, 0.0005663
7: -0.0000717, 0.0025678, -0.0000955, 0.0024057, -0.0010438, 0.0012425
8: -0.0005094, -0.0001671, -0.0004751, -0.0001611, -0.0001969, 0.0001568
9: 1.0024908, 1.0093082, 1.0023994, 1.0087383, -0.0023710, 0.0030465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 8

Time for candidate selection: 11.49 seconds

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0007851, upper bound: 0.0019065
time: 0.50 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0007140, upper bound: 0.0019227
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026498, -0.0011931, -0.0026498, -0.0011931, -0.0005650, 0.0005650
1: -0.0030310, 0.0009132, -0.0030310, 0.0009132, -0.0012382, 0.0012382
2: 0.0040830, 0.0074148, 0.0040830, 0.0074148, -0.0011214, 0.0011214
3: -0.0042551, -0.0038796, -0.0042551, -0.0038796, -0.0001993, 0.0001993
4: 0.0036870, 0.0063511, 0.0036870, 0.0063511, -0.0009909, 0.0009909
5: -0.0015522, 0.0011831, -0.0015522, 0.0011831, -0.0009376, 0.0009376
6: -0.0058807, -0.0044221, -0.0058807, -0.0044221, -0.0005785, 0.0005785
7: -0.0000717, 0.0025678, -0.0000717, 0.0025678, -0.0010817, 0.0010817
8: -0.0005094, -0.0001671, -0.0005094, -0.0001671, -0.0001593, 0.0001593
9: 1.0024908, 1.0093082, 1.0024908, 1.0093082, -0.0022898, 0.0022898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0014065, upper bound: 0.0020125
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0015179, upper bound: 0.0020616
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8

Time for candidate selection: 12.54 seconds

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0018601, upper bound: 0.0019287
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0018212, upper bound: 0.0019449
time: 0.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.92 seconds
IS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 14.92
Output dim: 9, lower bound: -0.0007851, upper bound: 0.0019065
IS_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 14.92
Output dim: 9, lower bound: -0.0007140, upper bound: 0.0019227
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.92
Output dim: 9, lower bound: -0.0018601, upper bound: 0.0019287
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 14.92
Output dim: 9, lower bound: -0.0018212, upper bound: 0.0019449

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.95 + 44.85 = 47.80 seconds
