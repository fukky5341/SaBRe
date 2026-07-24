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
Threshold: 0.00048692


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0032527, 0.0043265, 0.0032527, 0.0043265, -0.0005654, 0.0005654)
1: (0.0017922, 0.0019474, 0.0017922, 0.0019474, -0.0000817, 0.0000817)
2: (0.0119679, 0.0125616, 0.0119679, 0.0125616, -0.0003126, 0.0003126)
3: (-0.0023027, -0.0016887, -0.0023027, -0.0016887, -0.0003233, 0.0003233)
4: (-0.0022088, -0.0015441, -0.0022088, -0.0015441, -0.0003500, 0.0003500)
5: (0.0055746, 0.0062036, 0.0055746, 0.0062036, -0.0003312, 0.0003312)
6: (-0.0001820, 0.0023139, -0.0001820, 0.0023139, -0.0013142, 0.0013142)
7: (-0.0057081, -0.0023089, -0.0057081, -0.0023089, -0.0017898, 0.0017898)
8: (0.9851930, 0.9875875, 0.9851930, 0.9875875, -0.0012608, 0.0012608)
9: (-0.0046200, -0.0024465, -0.0046200, -0.0024465, -0.0011444, 0.0011444)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 1.35 = 2.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0007328, upper bound: 0.0007327

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006788, upper bound: 0.0006859
time: 0.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006858, upper bound: 0.0006858
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.24 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 8, lower bound: -0.0006788, upper bound: 0.0006859
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 8, lower bound: -0.0006858, upper bound: 0.0006858

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0032924, 0.0043259, 0.0032650, 0.0043263, -0.0005077, 0.0005457
1: 0.0017980, 0.0019473, 0.0017940, 0.0019473, -0.0000733, 0.0000788
2: 0.0119682, 0.0125396, 0.0119680, 0.0125547, -0.0003017, 0.0002807
3: -0.0023024, -0.0017114, -0.0023026, -0.0016958, -0.0003120, 0.0002903
4: -0.0021843, -0.0015445, -0.0022012, -0.0015442, -0.0003143, 0.0003378
5: 0.0055750, 0.0061804, 0.0055747, 0.0061964, -0.0003196, 0.0002974
6: -0.0001805, 0.0022216, -0.0001816, 0.0022852, -0.0012683, 0.0011800
7: -0.0055824, -0.0023109, -0.0056690, -0.0023094, -0.0016070, 0.0017273
8: 0.9852816, 0.9875861, 0.9852204, 0.9875870, -0.0011320, 0.0012167
9: -0.0046187, -0.0025268, -0.0046196, -0.0024714, -0.0011045, 0.0010276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006472, upper bound: 0.0006277
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006485, upper bound: 0.0006553
time: 0.57 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0033293, 0.0043786, 0.0032908, 0.0043262, -0.0005067, 0.0006640
1: 0.0018033, 0.0019549, 0.0017977, 0.0019473, -0.0000732, 0.0000959
2: 0.0119391, 0.0125192, 0.0119680, 0.0125405, -0.0003671, 0.0002801
3: -0.0023325, -0.0017326, -0.0023026, -0.0017105, -0.0003797, 0.0002897
4: -0.0021614, -0.0015119, -0.0021852, -0.0015443, -0.0003137, 0.0004110
5: 0.0055441, 0.0061587, 0.0055747, 0.0061813, -0.0003890, 0.0002968
6: -0.0003030, 0.0021357, -0.0001814, 0.0022253, -0.0015434, 0.0011777
7: -0.0054654, -0.0021441, -0.0055874, -0.0023097, -0.0016040, 0.0021020
8: 0.9853640, 0.9877036, 0.9852780, 0.9875869, -0.0011299, 0.0014807
9: -0.0047254, -0.0026016, -0.0046195, -0.0025236, -0.0013440, 0.0010256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006540, upper bound: 0.0006277
time: 0.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006553, upper bound: 0.0006553
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.50 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 8, lower bound: -0.0006472, upper bound: 0.0006277
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 8, lower bound: -0.0006485, upper bound: 0.0006553
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 8, lower bound: -0.0006540, upper bound: 0.0006277
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 8, lower bound: -0.0006553, upper bound: 0.0006553

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033332, 0.0043257, 0.0033511, 0.0043192, -0.0003997, 0.0004360
1: 0.0018039, 0.0019472, 0.0018064, 0.0019463, -0.0000577, 0.0000630
2: 0.0119683, 0.0125170, 0.0119719, 0.0125071, -0.0002411, 0.0002210
3: -0.0023023, -0.0017348, -0.0022986, -0.0017450, -0.0002493, 0.0002285
4: -0.0021590, -0.0015446, -0.0021479, -0.0015486, -0.0002474, 0.0002699
5: 0.0055751, 0.0061565, 0.0055789, 0.0061460, -0.0002554, 0.0002341
6: -0.0001801, 0.0021267, -0.0001650, 0.0020852, -0.0010135, 0.0009290
7: -0.0054531, -0.0023114, -0.0053966, -0.0023320, -0.0012652, 0.0013803
8: 0.9853726, 0.9875857, 0.9854124, 0.9875712, -0.0008912, 0.0009723
9: -0.0046184, -0.0026095, -0.0046052, -0.0026456, -0.0008826, 0.0008090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006100, upper bound: 0.0005962
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006101, upper bound: 0.0005889
time: 0.56 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033038, 0.0043258, 0.0032925, 0.0043262, -0.0005052, 0.0004015
1: 0.0017996, 0.0019473, 0.0017980, 0.0019473, -0.0000730, 0.0000580
2: 0.0119682, 0.0125333, 0.0119680, 0.0125395, -0.0002220, 0.0002793
3: -0.0023023, -0.0017179, -0.0023026, -0.0017115, -0.0002296, 0.0002889
4: -0.0021772, -0.0015445, -0.0021842, -0.0015443, -0.0003128, 0.0002485
5: 0.0055750, 0.0061737, 0.0055748, 0.0061803, -0.0002352, 0.0002960
6: -0.0001804, 0.0021952, -0.0001813, 0.0022213, -0.0009332, 0.0011743
7: -0.0055463, -0.0023111, -0.0055820, -0.0023099, -0.0015993, 0.0012710
8: 0.9853069, 0.9875859, 0.9852818, 0.9875867, -0.0011266, 0.0008953
9: -0.0046186, -0.0025499, -0.0046194, -0.0025271, -0.0008127, 0.0010227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006117, upper bound: 0.0006252
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006118, upper bound: 0.0006178
time: 0.57 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033698, 0.0043784, 0.0033778, 0.0043191, -0.0003990, 0.0005656
1: 0.0018091, 0.0019549, 0.0018103, 0.0019463, -0.0000576, 0.0000817
2: 0.0119392, 0.0124968, 0.0119719, 0.0124924, -0.0003127, 0.0002206
3: -0.0023324, -0.0017557, -0.0022985, -0.0017602, -0.0003234, 0.0002281
4: -0.0021363, -0.0015120, -0.0021314, -0.0015487, -0.0002470, 0.0003501
5: 0.0055442, 0.0061350, 0.0055789, 0.0061304, -0.0003313, 0.0002337
6: -0.0003026, 0.0020417, -0.0001648, 0.0020231, -0.0013145, 0.0009273
7: -0.0053373, -0.0021446, -0.0053121, -0.0023323, -0.0012629, 0.0017903
8: 0.9854541, 0.9877032, 0.9854720, 0.9875710, -0.0008896, 0.0012611
9: -0.0047250, -0.0026836, -0.0046050, -0.0026997, -0.0011448, 0.0008076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005962
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005889
time: 0.56 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033409, 0.0043785, 0.0033169, 0.0043261, -0.0005043, 0.0005562
1: 0.0018050, 0.0019549, 0.0018015, 0.0019473, -0.0000729, 0.0000803
2: 0.0119391, 0.0125128, 0.0119681, 0.0125260, -0.0003075, 0.0002788
3: -0.0023325, -0.0017391, -0.0023025, -0.0017254, -0.0003180, 0.0002884
4: -0.0021542, -0.0015119, -0.0021691, -0.0015444, -0.0003122, 0.0003443
5: 0.0055441, 0.0061520, 0.0055748, 0.0061660, -0.0003258, 0.0002954
6: -0.0003029, 0.0021089, -0.0001810, 0.0021646, -0.0012927, 0.0011722
7: -0.0054289, -0.0021442, -0.0055047, -0.0023101, -0.0015964, 0.0017605
8: 0.9853896, 0.9877034, 0.9853362, 0.9875866, -0.0011245, 0.0012401
9: -0.0047253, -0.0026250, -0.0046192, -0.0025765, -0.0011257, 0.0010208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006252
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006178, upper bound: 0.0006178
time: 0.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.64 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006100, upper bound: 0.0005962
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006101, upper bound: 0.0005889
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006117, upper bound: 0.0006252
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006118, upper bound: 0.0006178
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005962
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005889
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006252
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 8, lower bound: -0.0006178, upper bound: 0.0006178

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033357, 0.0043039, 0.0033517, 0.0043134, -0.0003870, 0.0004094
1: 0.0018042, 0.0019441, 0.0018065, 0.0019455, -0.0000559, 0.0000591
2: 0.0119803, 0.0125156, 0.0119751, 0.0125068, -0.0002263, 0.0002140
3: -0.0022898, -0.0017362, -0.0022952, -0.0017454, -0.0002341, 0.0002213
4: -0.0021574, -0.0015581, -0.0021475, -0.0015522, -0.0002396, 0.0002534
5: 0.0055878, 0.0061550, 0.0055823, 0.0061456, -0.0002398, 0.0002267
6: -0.0001295, 0.0021209, -0.0001515, 0.0020836, -0.0009515, 0.0008995
7: -0.0054452, -0.0023803, -0.0053944, -0.0023504, -0.0012250, 0.0012958
8: 0.9853781, 0.9875371, 0.9854139, 0.9875582, -0.0008629, 0.0009128
9: -0.0045743, -0.0026146, -0.0045935, -0.0026470, -0.0008286, 0.0007833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006100, upper bound: 0.0005880
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006100, upper bound: 0.0005962
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033061, 0.0042682, 0.0033528, 0.0042937, -0.0004654, 0.0004038
1: 0.0017999, 0.0019389, 0.0018067, 0.0019426, -0.0000672, 0.0000583
2: 0.0120001, 0.0125320, 0.0119860, 0.0125062, -0.0002232, 0.0002573
3: -0.0022694, -0.0017193, -0.0022840, -0.0017460, -0.0002309, 0.0002661
4: -0.0021757, -0.0015802, -0.0021468, -0.0015644, -0.0002881, 0.0002499
5: 0.0056088, 0.0061723, 0.0055938, 0.0061450, -0.0002365, 0.0002726
6: -0.0000464, 0.0021897, -0.0001058, 0.0020811, -0.0009384, 0.0010818
7: -0.0055388, -0.0024935, -0.0053909, -0.0024126, -0.0014733, 0.0012781
8: 0.9853122, 0.9874574, 0.9854164, 0.9875144, -0.0010378, 0.0009003
9: -0.0045019, -0.0025547, -0.0045537, -0.0026492, -0.0008172, 0.0009421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006101, upper bound: 0.0005808
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006101, upper bound: 0.0005889
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033063, 0.0043040, 0.0032932, 0.0043204, -0.0004954, 0.0003680
1: 0.0018000, 0.0019441, 0.0017981, 0.0019465, -0.0000716, 0.0000532
2: 0.0119803, 0.0125319, 0.0119712, 0.0125392, -0.0002034, 0.0002739
3: -0.0022899, -0.0017194, -0.0022993, -0.0017119, -0.0002104, 0.0002833
4: -0.0021756, -0.0015580, -0.0021838, -0.0015479, -0.0003067, 0.0002278
5: 0.0055878, 0.0061722, 0.0055782, 0.0061799, -0.0002156, 0.0002902
6: -0.0001298, 0.0021893, -0.0001678, 0.0022198, -0.0008553, 0.0011515
7: -0.0055383, -0.0023800, -0.0055798, -0.0023282, -0.0015683, 0.0011648
8: 0.9853126, 0.9875373, 0.9852834, 0.9875739, -0.0011047, 0.0008205
9: -0.0045745, -0.0025550, -0.0046076, -0.0025284, -0.0007448, 0.0010028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006117, upper bound: 0.0006150
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006117, upper bound: 0.0006253
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032782, 0.0042683, 0.0032943, 0.0043005, -0.0005446, 0.0003622
1: 0.0017959, 0.0019389, 0.0017982, 0.0019436, -0.0000787, 0.0000523
2: 0.0120001, 0.0125474, 0.0119822, 0.0125385, -0.0002003, 0.0003011
3: -0.0022694, -0.0017033, -0.0022879, -0.0017125, -0.0002071, 0.0003114
4: -0.0021930, -0.0015802, -0.0021831, -0.0015602, -0.0003371, 0.0002242
5: 0.0056087, 0.0061887, 0.0055898, 0.0061792, -0.0002122, 0.0003190
6: -0.0000466, 0.0022545, -0.0001216, 0.0022171, -0.0008419, 0.0012657
7: -0.0056272, -0.0024932, -0.0055762, -0.0023911, -0.0017238, 0.0011466
8: 0.9852499, 0.9874576, 0.9852858, 0.9875295, -0.0012143, 0.0008077
9: -0.0045021, -0.0024982, -0.0045674, -0.0025307, -0.0007332, 0.0011022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006118, upper bound: 0.0006090
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006118, upper bound: 0.0006178
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033722, 0.0043567, 0.0033785, 0.0043133, -0.0003863, 0.0005453
1: 0.0018095, 0.0019517, 0.0018104, 0.0019454, -0.0000558, 0.0000788
2: 0.0119512, 0.0124954, 0.0119752, 0.0124920, -0.0003015, 0.0002136
3: -0.0023200, -0.0017571, -0.0022952, -0.0017606, -0.0003118, 0.0002209
4: -0.0021348, -0.0015254, -0.0021310, -0.0015523, -0.0002391, 0.0003376
5: 0.0055569, 0.0061336, 0.0055823, 0.0061300, -0.0003195, 0.0002263
6: -0.0002521, 0.0020360, -0.0001513, 0.0020215, -0.0012675, 0.0008979
7: -0.0053296, -0.0022133, -0.0053099, -0.0023507, -0.0012229, 0.0017262
8: 0.9854596, 0.9876549, 0.9854735, 0.9875579, -0.0008615, 0.0012160
9: -0.0046811, -0.0026885, -0.0045933, -0.0027011, -0.0011038, 0.0007820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005880
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005880
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033463, 0.0043193, 0.0033796, 0.0042936, -0.0004651, 0.0005373
1: 0.0018057, 0.0019463, 0.0018105, 0.0019426, -0.0000672, 0.0000776
2: 0.0119718, 0.0125098, 0.0119860, 0.0124914, -0.0002971, 0.0002571
3: -0.0022986, -0.0017423, -0.0022839, -0.0017613, -0.0003073, 0.0002659
4: -0.0021508, -0.0015486, -0.0021303, -0.0015645, -0.0002879, 0.0003326
5: 0.0055788, 0.0061488, 0.0055938, 0.0061293, -0.0003148, 0.0002724
6: -0.0001652, 0.0020962, -0.0001056, 0.0020190, -0.0012489, 0.0010810
7: -0.0054116, -0.0023317, -0.0053064, -0.0024129, -0.0014722, 0.0017009
8: 0.9854018, 0.9875714, 0.9854759, 0.9875143, -0.0010371, 0.0011982
9: -0.0046054, -0.0026361, -0.0045535, -0.0027033, -0.0010876, 0.0009414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005808
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005807
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033434, 0.0043568, 0.0033176, 0.0043203, -0.0004945, 0.0005360
1: 0.0018053, 0.0019517, 0.0018016, 0.0019465, -0.0000714, 0.0000774
2: 0.0119511, 0.0125114, 0.0119713, 0.0125257, -0.0002963, 0.0002734
3: -0.0023201, -0.0017406, -0.0022992, -0.0017258, -0.0003065, 0.0002828
4: -0.0021527, -0.0015253, -0.0021686, -0.0015479, -0.0003061, 0.0003318
5: 0.0055568, 0.0061505, 0.0055782, 0.0061656, -0.0003140, 0.0002897
6: -0.0002524, 0.0021031, -0.0001676, 0.0021630, -0.0012458, 0.0011494
7: -0.0054210, -0.0022130, -0.0055025, -0.0023284, -0.0015653, 0.0016966
8: 0.9853952, 0.9876550, 0.9853378, 0.9875737, -0.0011026, 0.0011951
9: -0.0046813, -0.0026300, -0.0046075, -0.0025779, -0.0010849, 0.0010009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006150
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006150
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033160, 0.0043194, 0.0033188, 0.0043004, -0.0005437, 0.0005276
1: 0.0018014, 0.0019463, 0.0018018, 0.0019436, -0.0000785, 0.0000762
2: 0.0119718, 0.0125265, 0.0119823, 0.0125250, -0.0002917, 0.0003006
3: -0.0022987, -0.0017249, -0.0022878, -0.0017265, -0.0003017, 0.0003109
4: -0.0021696, -0.0015485, -0.0021679, -0.0015602, -0.0003365, 0.0003266
5: 0.0055787, 0.0061665, 0.0055899, 0.0061649, -0.0003091, 0.0003185
6: -0.0001655, 0.0021667, -0.0001214, 0.0021603, -0.0012264, 0.0012636
7: -0.0055075, -0.0023313, -0.0054989, -0.0023914, -0.0017209, 0.0016702
8: 0.9853343, 0.9875717, 0.9853404, 0.9875293, -0.0012123, 0.0011765
9: -0.0046057, -0.0025747, -0.0045672, -0.0025802, -0.0010680, 0.0011004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006090
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006090
time: 0.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.56 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006100, upper bound: 0.0005880
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006100, upper bound: 0.0005962
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006101, upper bound: 0.0005808
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006101, upper bound: 0.0005889
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006117, upper bound: 0.0006150
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006117, upper bound: 0.0006253
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006118, upper bound: 0.0006090
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006118, upper bound: 0.0006178
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005880
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005880
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005808
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006158, upper bound: 0.0005807
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006150
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006150
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006090
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.56
Output dim: 8, lower bound: -0.0006177, upper bound: 0.0006090

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033357, 0.0043039, 0.0033776, 0.0043129, -0.0003867, 0.0003669
1: 0.0018042, 0.0019441, 0.0018103, 0.0019454, -0.0000559, 0.0000530
2: 0.0119803, 0.0125156, 0.0119754, 0.0124925, -0.0002028, 0.0002138
3: -0.0022898, -0.0017362, -0.0022950, -0.0017601, -0.0002098, 0.0002211
4: -0.0021574, -0.0015581, -0.0021315, -0.0015525, -0.0002394, 0.0002271
5: 0.0055878, 0.0061550, 0.0055826, 0.0061305, -0.0002149, 0.0002266
6: -0.0001295, 0.0021209, -0.0001504, 0.0020236, -0.0008527, 0.0008989
7: -0.0054452, -0.0023803, -0.0053127, -0.0023519, -0.0012242, 0.0011613
8: 0.9853781, 0.9875371, 0.9854715, 0.9875572, -0.0008624, 0.0008180
9: -0.0045743, -0.0026146, -0.0045925, -0.0026993, -0.0007426, 0.0007828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005880
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005880
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033357, 0.0043039, 0.0034156, 0.0043632, -0.0005307, 0.0004235
1: 0.0018042, 0.0019441, 0.0018158, 0.0019527, -0.0000767, 0.0000612
2: 0.0119803, 0.0125156, 0.0119475, 0.0124715, -0.0002341, 0.0002934
3: -0.0022898, -0.0017362, -0.0023238, -0.0017819, -0.0002422, 0.0003035
4: -0.0021574, -0.0015581, -0.0021080, -0.0015214, -0.0003285, 0.0002622
5: 0.0055878, 0.0061550, 0.0055531, 0.0061082, -0.0002481, 0.0003109
6: -0.0001295, 0.0021209, -0.0002674, 0.0019353, -0.0009843, 0.0012336
7: -0.0054452, -0.0023803, -0.0051924, -0.0021925, -0.0016801, 0.0013406
8: 0.9853781, 0.9875371, 0.9855563, 0.9876693, -0.0011835, 0.0009443
9: -0.0045743, -0.0026146, -0.0046944, -0.0027762, -0.0008572, 0.0010743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005962
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005962
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033061, 0.0042682, 0.0033786, 0.0042932, -0.0004651, 0.0003613
1: 0.0017999, 0.0019389, 0.0018104, 0.0019425, -0.0000672, 0.0000522
2: 0.0120001, 0.0125320, 0.0119862, 0.0124919, -0.0001998, 0.0002572
3: -0.0022694, -0.0017193, -0.0022837, -0.0017607, -0.0002066, 0.0002660
4: -0.0021757, -0.0015802, -0.0021308, -0.0015647, -0.0002879, 0.0002237
5: 0.0056088, 0.0061723, 0.0055941, 0.0061298, -0.0002117, 0.0002725
6: -0.0000464, 0.0021897, -0.0001046, 0.0020211, -0.0008399, 0.0010811
7: -0.0055388, -0.0024935, -0.0053093, -0.0024142, -0.0014723, 0.0011438
8: 0.9853122, 0.9874574, 0.9854739, 0.9875132, -0.0010371, 0.0008057
9: -0.0045019, -0.0025547, -0.0045526, -0.0027014, -0.0007314, 0.0009414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005808
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005808
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033061, 0.0042682, 0.0034168, 0.0043418, -0.0005849, 0.0004178
1: 0.0017999, 0.0019389, 0.0018159, 0.0019496, -0.0000845, 0.0000604
2: 0.0120001, 0.0125320, 0.0119594, 0.0124708, -0.0002310, 0.0003234
3: -0.0022694, -0.0017193, -0.0023115, -0.0017825, -0.0002389, 0.0003344
4: -0.0021757, -0.0015802, -0.0021072, -0.0015346, -0.0003620, 0.0002586
5: 0.0056088, 0.0061723, 0.0055656, 0.0061075, -0.0002447, 0.0003426
6: -0.0000464, 0.0021897, -0.0002176, 0.0019325, -0.0009711, 0.0013594
7: -0.0055388, -0.0024935, -0.0051886, -0.0022604, -0.0018513, 0.0013225
8: 0.9853122, 0.9874574, 0.9855589, 0.9876217, -0.0013041, 0.0009316
9: -0.0045019, -0.0025547, -0.0046510, -0.0027786, -0.0008457, 0.0011838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005889
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005889
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033063, 0.0043040, 0.0033209, 0.0043199, -0.0004952, 0.0003256
1: 0.0018000, 0.0019441, 0.0018021, 0.0019464, -0.0000715, 0.0000470
2: 0.0119803, 0.0125319, 0.0119715, 0.0125238, -0.0001800, 0.0002738
3: -0.0022899, -0.0017194, -0.0022990, -0.0017278, -0.0001862, 0.0002831
4: -0.0021756, -0.0015580, -0.0021666, -0.0015482, -0.0003065, 0.0002016
5: 0.0055878, 0.0061722, 0.0055784, 0.0061636, -0.0001907, 0.0002901
6: -0.0001298, 0.0021893, -0.0001667, 0.0021552, -0.0007568, 0.0011509
7: -0.0055383, -0.0023800, -0.0054919, -0.0023297, -0.0015675, 0.0010307
8: 0.9853126, 0.9875373, 0.9853452, 0.9875728, -0.0011041, 0.0007261
9: -0.0045745, -0.0025550, -0.0046067, -0.0025847, -0.0006591, 0.0010023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006132
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005901
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033063, 0.0043040, 0.0033584, 0.0043726, -0.0006105, 0.0003807
1: 0.0018000, 0.0019441, 0.0018075, 0.0019540, -0.0000882, 0.0000550
2: 0.0119803, 0.0125319, 0.0119423, 0.0125031, -0.0002105, 0.0003375
3: -0.0022899, -0.0017194, -0.0023291, -0.0017492, -0.0002177, 0.0003491
4: -0.0021756, -0.0015580, -0.0021434, -0.0015155, -0.0003779, 0.0002356
5: 0.0055878, 0.0061722, 0.0055476, 0.0061417, -0.0002230, 0.0003576
6: -0.0001298, 0.0021893, -0.0002892, 0.0020682, -0.0008847, 0.0014190
7: -0.0055383, -0.0023800, -0.0053734, -0.0021628, -0.0019325, 0.0012049
8: 0.9853126, 0.9875373, 0.9854288, 0.9876904, -0.0013613, 0.0008488
9: -0.0045745, -0.0025550, -0.0047134, -0.0026605, -0.0007705, 0.0012357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006234
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005986
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032782, 0.0042683, 0.0033221, 0.0043000, -0.0005443, 0.0003199
1: 0.0017959, 0.0019389, 0.0018022, 0.0019435, -0.0000786, 0.0000462
2: 0.0120001, 0.0125474, 0.0119825, 0.0125232, -0.0001769, 0.0003009
3: -0.0022694, -0.0017033, -0.0022876, -0.0017284, -0.0001829, 0.0003112
4: -0.0021930, -0.0015802, -0.0021659, -0.0015605, -0.0003369, 0.0001980
5: 0.0056087, 0.0061887, 0.0055901, 0.0061630, -0.0001874, 0.0003189
6: -0.0000466, 0.0022545, -0.0001205, 0.0021525, -0.0007435, 0.0012651
7: -0.0056272, -0.0024932, -0.0054883, -0.0023927, -0.0017230, 0.0010126
8: 0.9852499, 0.9874576, 0.9853477, 0.9875284, -0.0012137, 0.0007133
9: -0.0045021, -0.0024982, -0.0045664, -0.0025870, -0.0006475, 0.0011017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006072
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005830
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032782, 0.0042683, 0.0033596, 0.0043522, -0.0006508, 0.0003749
1: 0.0017959, 0.0019389, 0.0018077, 0.0019511, -0.0000940, 0.0000542
2: 0.0120001, 0.0125474, 0.0119536, 0.0125024, -0.0002073, 0.0003598
3: -0.0022694, -0.0017033, -0.0023175, -0.0017499, -0.0002144, 0.0003721
4: -0.0021930, -0.0015802, -0.0021426, -0.0015282, -0.0004029, 0.0002320
5: 0.0056087, 0.0061887, 0.0055595, 0.0061410, -0.0002196, 0.0003812
6: -0.0000466, 0.0022545, -0.0002418, 0.0020653, -0.0008713, 0.0015126
7: -0.0056272, -0.0024932, -0.0053694, -0.0022274, -0.0020601, 0.0011866
8: 0.9852499, 0.9874576, 0.9854315, 0.9876449, -0.0014511, 0.0008359
9: -0.0045021, -0.0024982, -0.0046721, -0.0026630, -0.0007588, 0.0013173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006158
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005914
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033722, 0.0043567, 0.0033776, 0.0043129, -0.0004347, 0.0004868
1: 0.0018095, 0.0019517, 0.0018103, 0.0019454, -0.0000628, 0.0000703
2: 0.0119512, 0.0124954, 0.0119754, 0.0124925, -0.0002692, 0.0002403
3: -0.0023200, -0.0017571, -0.0022950, -0.0017601, -0.0002784, 0.0002486
4: -0.0021348, -0.0015254, -0.0021315, -0.0015525, -0.0002691, 0.0003014
5: 0.0055569, 0.0061336, 0.0055826, 0.0061305, -0.0002852, 0.0002546
6: -0.0002521, 0.0020360, -0.0001504, 0.0020236, -0.0011315, 0.0010103
7: -0.0053296, -0.0022133, -0.0053127, -0.0023519, -0.0013759, 0.0015410
8: 0.9854596, 0.9876549, 0.9854715, 0.9875572, -0.0009692, 0.0010855
9: -0.0046811, -0.0026885, -0.0045925, -0.0026993, -0.0009854, 0.0008798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033722, 0.0043567, 0.0034156, 0.0043632, -0.0003895, 0.0003714
1: 0.0018095, 0.0019517, 0.0018158, 0.0019527, -0.0000563, 0.0000537
2: 0.0119512, 0.0124954, 0.0119475, 0.0124715, -0.0002054, 0.0002154
3: -0.0023200, -0.0017571, -0.0023238, -0.0017819, -0.0002124, 0.0002227
4: -0.0021348, -0.0015254, -0.0021080, -0.0015214, -0.0002411, 0.0002299
5: 0.0055569, 0.0061336, 0.0055531, 0.0061082, -0.0002176, 0.0002282
6: -0.0002521, 0.0020360, -0.0002674, 0.0019353, -0.0008633, 0.0009054
7: -0.0053296, -0.0022133, -0.0051924, -0.0021925, -0.0012330, 0.0011758
8: 0.9854596, 0.9876549, 0.9855563, 0.9876693, -0.0008686, 0.0008283
9: -0.0046811, -0.0026885, -0.0046944, -0.0027762, -0.0007518, 0.0007884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033463, 0.0043193, 0.0033786, 0.0042932, -0.0004744, 0.0004789
1: 0.0018057, 0.0019463, 0.0018104, 0.0019425, -0.0000685, 0.0000692
2: 0.0119718, 0.0125098, 0.0119862, 0.0124919, -0.0002648, 0.0002623
3: -0.0022986, -0.0017423, -0.0022837, -0.0017607, -0.0002738, 0.0002713
4: -0.0021508, -0.0015486, -0.0021308, -0.0015647, -0.0002937, 0.0002965
5: 0.0055788, 0.0061488, 0.0055941, 0.0061298, -0.0002805, 0.0002779
6: -0.0001652, 0.0020962, -0.0001046, 0.0020211, -0.0011131, 0.0011027
7: -0.0054116, -0.0023317, -0.0053093, -0.0024142, -0.0015017, 0.0015160
8: 0.9854018, 0.9875714, 0.9854739, 0.9875132, -0.0010578, 0.0010679
9: -0.0046054, -0.0026361, -0.0045526, -0.0027014, -0.0009694, 0.0009602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033463, 0.0043193, 0.0034168, 0.0043418, -0.0004677, 0.0003639
1: 0.0018057, 0.0019463, 0.0018159, 0.0019496, -0.0000676, 0.0000526
2: 0.0119718, 0.0125098, 0.0119594, 0.0124708, -0.0002012, 0.0002586
3: -0.0022986, -0.0017423, -0.0023115, -0.0017825, -0.0002081, 0.0002674
4: -0.0021508, -0.0015486, -0.0021072, -0.0015346, -0.0002895, 0.0002253
5: 0.0055788, 0.0061488, 0.0055656, 0.0061075, -0.0002132, 0.0002740
6: -0.0001652, 0.0020962, -0.0002176, 0.0019325, -0.0008459, 0.0010870
7: -0.0054116, -0.0023317, -0.0051886, -0.0022604, -0.0014805, 0.0011520
8: 0.9854018, 0.9875714, 0.9855589, 0.9876217, -0.0010429, 0.0008115
9: -0.0046054, -0.0026361, -0.0046510, -0.0027786, -0.0007366, 0.0009466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005807
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005807
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033434, 0.0043568, 0.0033209, 0.0043199, -0.0005255, 0.0004785
1: 0.0018053, 0.0019517, 0.0018021, 0.0019464, -0.0000759, 0.0000691
2: 0.0119511, 0.0125114, 0.0119715, 0.0125238, -0.0002645, 0.0002906
3: -0.0023201, -0.0017406, -0.0022990, -0.0017278, -0.0002736, 0.0003005
4: -0.0021527, -0.0015253, -0.0021666, -0.0015482, -0.0003253, 0.0002962
5: 0.0055568, 0.0061505, 0.0055784, 0.0061636, -0.0002803, 0.0003079
6: -0.0002524, 0.0021031, -0.0001667, 0.0021552, -0.0011121, 0.0012215
7: -0.0054210, -0.0022130, -0.0054919, -0.0023297, -0.0016636, 0.0015145
8: 0.9853952, 0.9876550, 0.9853452, 0.9875728, -0.0011719, 0.0010669
9: -0.0046813, -0.0026300, -0.0046067, -0.0025847, -0.0009684, 0.0010638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006132
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005900
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033434, 0.0043568, 0.0033584, 0.0043726, -0.0004987, 0.0003316
1: 0.0018053, 0.0019517, 0.0018075, 0.0019540, -0.0000720, 0.0000479
2: 0.0119511, 0.0125114, 0.0119423, 0.0125031, -0.0001833, 0.0002757
3: -0.0023201, -0.0017406, -0.0023291, -0.0017492, -0.0001896, 0.0002852
4: -0.0021527, -0.0015253, -0.0021434, -0.0015155, -0.0003087, 0.0002053
5: 0.0055568, 0.0061505, 0.0055476, 0.0061417, -0.0001943, 0.0002921
6: -0.0002524, 0.0021031, -0.0002892, 0.0020682, -0.0007708, 0.0011591
7: -0.0054210, -0.0022130, -0.0053734, -0.0021628, -0.0015786, 0.0010497
8: 0.9853952, 0.9876550, 0.9854288, 0.9876904, -0.0011120, 0.0007394
9: -0.0046813, -0.0026300, -0.0047134, -0.0026605, -0.0006712, 0.0010094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006133
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005901
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033160, 0.0043194, 0.0033221, 0.0043000, -0.0005495, 0.0004701
1: 0.0018014, 0.0019463, 0.0018022, 0.0019435, -0.0000794, 0.0000679
2: 0.0119718, 0.0125265, 0.0119825, 0.0125232, -0.0002599, 0.0003038
3: -0.0022987, -0.0017249, -0.0022876, -0.0017284, -0.0002688, 0.0003142
4: -0.0021696, -0.0015485, -0.0021659, -0.0015605, -0.0003401, 0.0002910
5: 0.0055787, 0.0061665, 0.0055901, 0.0061630, -0.0002754, 0.0003219
6: -0.0001655, 0.0021667, -0.0001205, 0.0021525, -0.0010927, 0.0012771
7: -0.0055075, -0.0023313, -0.0054883, -0.0023927, -0.0017393, 0.0014882
8: 0.9853343, 0.9875717, 0.9853477, 0.9875284, -0.0012252, 0.0010483
9: -0.0046057, -0.0025747, -0.0045664, -0.0025870, -0.0009516, 0.0011122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006072
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005829
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033160, 0.0043194, 0.0033596, 0.0043522, -0.0005472, 0.0003249
1: 0.0018014, 0.0019463, 0.0018077, 0.0019511, -0.0000790, 0.0000469
2: 0.0119718, 0.0125265, 0.0119536, 0.0125024, -0.0001796, 0.0003025
3: -0.0022987, -0.0017249, -0.0023175, -0.0017499, -0.0001858, 0.0003129
4: -0.0021696, -0.0015485, -0.0021426, -0.0015282, -0.0003387, 0.0002011
5: 0.0055787, 0.0061665, 0.0055595, 0.0061410, -0.0001903, 0.0003205
6: -0.0001655, 0.0021667, -0.0002418, 0.0020653, -0.0007551, 0.0012717
7: -0.0055075, -0.0023313, -0.0053694, -0.0022274, -0.0017320, 0.0010284
8: 0.9853343, 0.9875717, 0.9854315, 0.9876449, -0.0012201, 0.0007244
9: -0.0046057, -0.0025747, -0.0046721, -0.0026630, -0.0006576, 0.0011075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006072
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005830
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.67 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005880
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005880
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005962
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005962
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005808
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005808
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005889
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005889
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006132
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005901
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006234
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005986
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006072
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005830
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006158
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005914
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005807
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005807
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006132
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005900
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006133
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005901
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006072
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005829
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006072
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005830

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033776, 0.0043129, -0.0003292, 0.0003076
1: 0.0018105, 0.0019431, 0.0018103, 0.0019454, -0.0000476, 0.0000444
2: 0.0119841, 0.0124915, 0.0119754, 0.0124925, -0.0001701, 0.0001820
3: -0.0022859, -0.0017612, -0.0022950, -0.0017601, -0.0001759, 0.0001882
4: -0.0021303, -0.0015624, -0.0021315, -0.0015525, -0.0002038, 0.0001904
5: 0.0055919, 0.0061294, 0.0055826, 0.0061305, -0.0001802, 0.0001928
6: -0.0001135, 0.0020192, -0.0001504, 0.0020236, -0.0007149, 0.0007651
7: -0.0053067, -0.0024022, -0.0053127, -0.0023519, -0.0010420, 0.0009737
8: 0.9854757, 0.9875218, 0.9854715, 0.9875572, -0.0007340, 0.0006859
9: -0.0045603, -0.0027031, -0.0045925, -0.0026993, -0.0006226, 0.0006663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005933
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005933
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033776, 0.0043129, -0.0004377, 0.0003669
1: 0.0018024, 0.0019441, 0.0018103, 0.0019454, -0.0000632, 0.0000530
2: 0.0119803, 0.0125228, 0.0119754, 0.0124925, -0.0002028, 0.0002420
3: -0.0022898, -0.0017288, -0.0022950, -0.0017601, -0.0002098, 0.0002503
4: -0.0021654, -0.0015581, -0.0021315, -0.0015525, -0.0002710, 0.0002271
5: 0.0055878, 0.0061625, 0.0055826, 0.0061305, -0.0002149, 0.0002564
6: -0.0001295, 0.0021509, -0.0001504, 0.0020236, -0.0008527, 0.0010174
7: -0.0054860, -0.0023803, -0.0053127, -0.0023519, -0.0013856, 0.0011613
8: 0.9853494, 0.9875371, 0.9854715, 0.9875572, -0.0009760, 0.0008181
9: -0.0045743, -0.0025884, -0.0045925, -0.0026993, -0.0007426, 0.0008860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005933
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005933
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0034156, 0.0043632, -0.0004732, 0.0003642
1: 0.0018105, 0.0019431, 0.0018158, 0.0019527, -0.0000684, 0.0000526
2: 0.0119841, 0.0124915, 0.0119475, 0.0124715, -0.0002014, 0.0002616
3: -0.0022859, -0.0017612, -0.0023238, -0.0017819, -0.0002083, 0.0002706
4: -0.0021303, -0.0015624, -0.0021080, -0.0015214, -0.0002929, 0.0002255
5: 0.0055919, 0.0061294, 0.0055531, 0.0061082, -0.0002134, 0.0002772
6: -0.0001135, 0.0020192, -0.0002674, 0.0019353, -0.0008466, 0.0010998
7: -0.0053067, -0.0024022, -0.0051924, -0.0021925, -0.0014978, 0.0011530
8: 0.9854757, 0.9875218, 0.9855563, 0.9876693, -0.0010551, 0.0008122
9: -0.0045603, -0.0027031, -0.0046944, -0.0027762, -0.0007372, 0.0009578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005962
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005962
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0034156, 0.0043632, -0.0005817, 0.0004235
1: 0.0018024, 0.0019441, 0.0018158, 0.0019527, -0.0000840, 0.0000612
2: 0.0119803, 0.0125228, 0.0119475, 0.0124715, -0.0002341, 0.0003216
3: -0.0022898, -0.0017288, -0.0023238, -0.0017819, -0.0002422, 0.0003326
4: -0.0021654, -0.0015581, -0.0021080, -0.0015214, -0.0003601, 0.0002622
5: 0.0055878, 0.0061625, 0.0055531, 0.0061082, -0.0002481, 0.0003408
6: -0.0001295, 0.0021509, -0.0002674, 0.0019353, -0.0009844, 0.0013521
7: -0.0054860, -0.0023803, -0.0051924, -0.0021925, -0.0018414, 0.0013406
8: 0.9853494, 0.9875371, 0.9855563, 0.9876693, -0.0012971, 0.0009444
9: -0.0045743, -0.0025884, -0.0046944, -0.0027762, -0.0008572, 0.0011775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005962
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005962
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033786, 0.0042932, -0.0004224, 0.0003031
1: 0.0018062, 0.0019379, 0.0018104, 0.0019425, -0.0000610, 0.0000438
2: 0.0120039, 0.0125079, 0.0119862, 0.0124919, -0.0001676, 0.0002335
3: -0.0022654, -0.0017442, -0.0022837, -0.0017607, -0.0001733, 0.0002415
4: -0.0021488, -0.0015845, -0.0021308, -0.0015647, -0.0002615, 0.0001876
5: 0.0056128, 0.0061468, 0.0055941, 0.0061298, -0.0001776, 0.0002474
6: -0.0000304, 0.0020885, -0.0001046, 0.0020211, -0.0007045, 0.0009817
7: -0.0054010, -0.0025153, -0.0053093, -0.0024142, -0.0013370, 0.0009595
8: 0.9854093, 0.9874420, 0.9854739, 0.9875132, -0.0009418, 0.0006759
9: -0.0044880, -0.0026428, -0.0045526, -0.0027014, -0.0006135, 0.0008549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005896
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005896
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033786, 0.0042932, -0.0004981, 0.0003613
1: 0.0017983, 0.0019389, 0.0018104, 0.0019425, -0.0000720, 0.0000522
2: 0.0120001, 0.0125384, 0.0119862, 0.0124919, -0.0001998, 0.0002754
3: -0.0022694, -0.0017127, -0.0022837, -0.0017607, -0.0002066, 0.0002848
4: -0.0021829, -0.0015802, -0.0021308, -0.0015647, -0.0003083, 0.0002237
5: 0.0056088, 0.0061791, 0.0055941, 0.0061298, -0.0002117, 0.0002918
6: -0.0000464, 0.0022166, -0.0001046, 0.0020211, -0.0008399, 0.0011578
7: -0.0055755, -0.0024935, -0.0053093, -0.0024142, -0.0015768, 0.0011438
8: 0.9852863, 0.9874574, 0.9854739, 0.9875132, -0.0011107, 0.0008057
9: -0.0045019, -0.0025312, -0.0045526, -0.0027014, -0.0007314, 0.0010082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005896
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005896
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0034168, 0.0043418, -0.0005421, 0.0003596
1: 0.0018062, 0.0019379, 0.0018159, 0.0019496, -0.0000783, 0.0000519
2: 0.0120039, 0.0125079, 0.0119594, 0.0124708, -0.0001988, 0.0002997
3: -0.0022654, -0.0017442, -0.0023115, -0.0017825, -0.0002056, 0.0003100
4: -0.0021488, -0.0015845, -0.0021072, -0.0015346, -0.0003356, 0.0002226
5: 0.0056128, 0.0061468, 0.0055656, 0.0061075, -0.0002106, 0.0003176
6: -0.0000304, 0.0020885, -0.0002176, 0.0019325, -0.0008358, 0.0012600
7: -0.0054010, -0.0025153, -0.0051886, -0.0022604, -0.0017160, 0.0011382
8: 0.9854093, 0.9874420, 0.9855589, 0.9876217, -0.0012088, 0.0008018
9: -0.0044880, -0.0026428, -0.0046510, -0.0027786, -0.0007278, 0.0010973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005888
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005889
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0034168, 0.0043418, -0.0006179, 0.0004178
1: 0.0017983, 0.0019389, 0.0018159, 0.0019496, -0.0000893, 0.0000604
2: 0.0120001, 0.0125384, 0.0119594, 0.0124708, -0.0002310, 0.0003416
3: -0.0022694, -0.0017127, -0.0023115, -0.0017825, -0.0002389, 0.0003533
4: -0.0021829, -0.0015802, -0.0021072, -0.0015346, -0.0003825, 0.0002586
5: 0.0056088, 0.0061791, 0.0055656, 0.0061075, -0.0002447, 0.0003619
6: -0.0000464, 0.0022166, -0.0002176, 0.0019325, -0.0009711, 0.0014361
7: -0.0055755, -0.0024935, -0.0051886, -0.0022604, -0.0019558, 0.0013225
8: 0.9852863, 0.9874574, 0.9855589, 0.9876217, -0.0013777, 0.0009316
9: -0.0045019, -0.0025312, -0.0046510, -0.0027786, -0.0008457, 0.0012506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005889
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005889
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033209, 0.0043199, -0.0003838, 0.0004162
1: 0.0018105, 0.0019431, 0.0018021, 0.0019464, -0.0000555, 0.0000601
2: 0.0119841, 0.0124915, 0.0119715, 0.0125238, -0.0002301, 0.0002122
3: -0.0022859, -0.0017612, -0.0022990, -0.0017278, -0.0002380, 0.0002195
4: -0.0021303, -0.0015624, -0.0021666, -0.0015482, -0.0002376, 0.0002576
5: 0.0055919, 0.0061294, 0.0055784, 0.0061636, -0.0002438, 0.0002248
6: -0.0001135, 0.0020192, -0.0001667, 0.0021552, -0.0009674, 0.0008921
7: -0.0053067, -0.0024022, -0.0054919, -0.0023297, -0.0012150, 0.0013175
8: 0.9854757, 0.9875218, 0.9853452, 0.9875728, -0.0008559, 0.0009281
9: -0.0045603, -0.0027031, -0.0046067, -0.0025847, -0.0008425, 0.0007769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006159
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006160
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033209, 0.0043199, -0.0003466, 0.0003256
1: 0.0018024, 0.0019441, 0.0018021, 0.0019464, -0.0000501, 0.0000470
2: 0.0119803, 0.0125228, 0.0119715, 0.0125238, -0.0001800, 0.0001916
3: -0.0022898, -0.0017288, -0.0022990, -0.0017278, -0.0001862, 0.0001982
4: -0.0021654, -0.0015581, -0.0021666, -0.0015482, -0.0002145, 0.0002015
5: 0.0055878, 0.0061625, 0.0055784, 0.0061636, -0.0001907, 0.0002030
6: -0.0001295, 0.0021509, -0.0001667, 0.0021552, -0.0007567, 0.0008056
7: -0.0054860, -0.0023803, -0.0054919, -0.0023297, -0.0010971, 0.0010306
8: 0.9853494, 0.9875371, 0.9853452, 0.9875728, -0.0007728, 0.0007260
9: -0.0045743, -0.0025884, -0.0046067, -0.0025847, -0.0006590, 0.0007015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005950
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005950
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033584, 0.0043726, -0.0004992, 0.0004385
1: 0.0018105, 0.0019431, 0.0018075, 0.0019540, -0.0000721, 0.0000634
2: 0.0119841, 0.0124915, 0.0119423, 0.0125031, -0.0002425, 0.0002760
3: -0.0022859, -0.0017612, -0.0023291, -0.0017492, -0.0002508, 0.0002854
4: -0.0021303, -0.0015624, -0.0021434, -0.0015155, -0.0003090, 0.0002715
5: 0.0055919, 0.0061294, 0.0055476, 0.0061417, -0.0002569, 0.0002924
6: -0.0001135, 0.0020192, -0.0002892, 0.0020682, -0.0010193, 0.0011602
7: -0.0053067, -0.0024022, -0.0053734, -0.0021628, -0.0015801, 0.0013882
8: 0.9854757, 0.9875218, 0.9854288, 0.9876904, -0.0011130, 0.0009778
9: -0.0045603, -0.0027031, -0.0047134, -0.0026605, -0.0008876, 0.0010103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0006234
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0006234
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033584, 0.0043726, -0.0004910, 0.0003806
1: 0.0018024, 0.0019441, 0.0018075, 0.0019540, -0.0000709, 0.0000550
2: 0.0119803, 0.0125228, 0.0119423, 0.0125031, -0.0002104, 0.0002715
3: -0.0022898, -0.0017288, -0.0023291, -0.0017492, -0.0002176, 0.0002808
4: -0.0021654, -0.0015581, -0.0021434, -0.0015155, -0.0003040, 0.0002356
5: 0.0055878, 0.0061625, 0.0055476, 0.0061417, -0.0002230, 0.0002876
6: -0.0001295, 0.0021509, -0.0002892, 0.0020682, -0.0008847, 0.0011413
7: -0.0054860, -0.0023803, -0.0053734, -0.0021628, -0.0015544, 0.0012048
8: 0.9853494, 0.9875371, 0.9854288, 0.9876904, -0.0010949, 0.0008487
9: -0.0045743, -0.0025884, -0.0047134, -0.0026605, -0.0007704, 0.0009939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005986
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005986
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033221, 0.0043000, -0.0004615, 0.0004116
1: 0.0018062, 0.0019379, 0.0018022, 0.0019435, -0.0000667, 0.0000595
2: 0.0120039, 0.0125079, 0.0119825, 0.0125232, -0.0002276, 0.0002552
3: -0.0022654, -0.0017442, -0.0022876, -0.0017284, -0.0002354, 0.0002639
4: -0.0021488, -0.0015845, -0.0021659, -0.0015605, -0.0002857, 0.0002548
5: 0.0056128, 0.0061468, 0.0055901, 0.0061630, -0.0002411, 0.0002704
6: -0.0000304, 0.0020885, -0.0001205, 0.0021525, -0.0009567, 0.0010727
7: -0.0054010, -0.0025153, -0.0054883, -0.0023927, -0.0014609, 0.0013030
8: 0.9854093, 0.9874420, 0.9853477, 0.9875284, -0.0010291, 0.0009178
9: -0.0044880, -0.0026428, -0.0045664, -0.0025870, -0.0008332, 0.0009341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006106
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006106
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033221, 0.0043000, -0.0004362, 0.0003199
1: 0.0017983, 0.0019389, 0.0018022, 0.0019435, -0.0000630, 0.0000462
2: 0.0120001, 0.0125384, 0.0119825, 0.0125232, -0.0001768, 0.0002412
3: -0.0022694, -0.0017127, -0.0022876, -0.0017284, -0.0001829, 0.0002494
4: -0.0021829, -0.0015802, -0.0021659, -0.0015605, -0.0002700, 0.0001980
5: 0.0056088, 0.0061791, 0.0055901, 0.0061630, -0.0001874, 0.0002555
6: -0.0000464, 0.0022166, -0.0001205, 0.0021525, -0.0007434, 0.0010138
7: -0.0055755, -0.0024935, -0.0054883, -0.0023927, -0.0013807, 0.0010125
8: 0.9852863, 0.9874574, 0.9853477, 0.9875284, -0.0009726, 0.0007132
9: -0.0045019, -0.0025312, -0.0045664, -0.0025870, -0.0006474, 0.0008829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005914
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005914
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033596, 0.0043522, -0.0005680, 0.0004339
1: 0.0018062, 0.0019379, 0.0018077, 0.0019511, -0.0000821, 0.0000627
2: 0.0120039, 0.0125079, 0.0119536, 0.0125024, -0.0002399, 0.0003140
3: -0.0022654, -0.0017442, -0.0023175, -0.0017499, -0.0002481, 0.0003248
4: -0.0021488, -0.0015845, -0.0021426, -0.0015282, -0.0003516, 0.0002686
5: 0.0056128, 0.0061468, 0.0055595, 0.0061410, -0.0002542, 0.0003327
6: -0.0000304, 0.0020885, -0.0002418, 0.0020653, -0.0010085, 0.0013202
7: -0.0054010, -0.0025153, -0.0053694, -0.0022274, -0.0017979, 0.0013736
8: 0.9854093, 0.9874420, 0.9854315, 0.9876449, -0.0012665, 0.0009676
9: -0.0044880, -0.0026428, -0.0046721, -0.0026630, -0.0008783, 0.0011496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0006158
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0006158
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033596, 0.0043522, -0.0005564, 0.0003748
1: 0.0017983, 0.0019389, 0.0018077, 0.0019511, -0.0000804, 0.0000542
2: 0.0120001, 0.0125384, 0.0119536, 0.0125024, -0.0002072, 0.0003076
3: -0.0022694, -0.0017127, -0.0023175, -0.0017499, -0.0002143, 0.0003181
4: -0.0021829, -0.0015802, -0.0021426, -0.0015282, -0.0003444, 0.0002320
5: 0.0056088, 0.0061791, 0.0055595, 0.0061410, -0.0002196, 0.0003259
6: -0.0000464, 0.0022166, -0.0002418, 0.0020653, -0.0008712, 0.0012931
7: -0.0055755, -0.0024935, -0.0053694, -0.0022274, -0.0017611, 0.0011865
8: 0.9852863, 0.9874574, 0.9854315, 0.9876449, -0.0012406, 0.0008358
9: -0.0045019, -0.0025312, -0.0046721, -0.0026630, -0.0007587, 0.0011261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005914
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005914
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033776, 0.0043129, -0.0003858, 0.0004605
1: 0.0018160, 0.0019505, 0.0018103, 0.0019454, -0.0000557, 0.0000665
2: 0.0119557, 0.0124705, 0.0119754, 0.0124925, -0.0002546, 0.0002133
3: -0.0023154, -0.0017829, -0.0022950, -0.0017601, -0.0002633, 0.0002206
4: -0.0021069, -0.0015304, -0.0021315, -0.0015525, -0.0002388, 0.0002851
5: 0.0055617, 0.0061072, 0.0055826, 0.0061305, -0.0002698, 0.0002260
6: -0.0002333, 0.0019311, -0.0001504, 0.0020236, -0.0010704, 0.0008967
7: -0.0051868, -0.0022390, -0.0053127, -0.0023519, -0.0012212, 0.0014578
8: 0.9855602, 0.9876367, 0.9854715, 0.9875572, -0.0008602, 0.0010269
9: -0.0046647, -0.0027798, -0.0045925, -0.0026993, -0.0009322, 0.0007808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005927
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005927
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033776, 0.0043129, -0.0004600, 0.0004868
1: 0.0018078, 0.0019517, 0.0018103, 0.0019454, -0.0000665, 0.0000703
2: 0.0119511, 0.0125021, 0.0119754, 0.0124925, -0.0002692, 0.0002543
3: -0.0023200, -0.0017502, -0.0022950, -0.0017601, -0.0002784, 0.0002630
4: -0.0021422, -0.0015254, -0.0021315, -0.0015525, -0.0002848, 0.0003014
5: 0.0055569, 0.0061406, 0.0055826, 0.0061305, -0.0002852, 0.0002695
6: -0.0002522, 0.0020639, -0.0001504, 0.0020236, -0.0011316, 0.0010692
7: -0.0053675, -0.0022132, -0.0053127, -0.0023519, -0.0014562, 0.0015411
8: 0.9854329, 0.9876548, 0.9854715, 0.9875572, -0.0010258, 0.0010856
9: -0.0046812, -0.0026642, -0.0045925, -0.0026993, -0.0009854, 0.0009311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005926
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005926
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0034156, 0.0043632, -0.0003326, 0.0003123
1: 0.0018160, 0.0019505, 0.0018158, 0.0019527, -0.0000481, 0.0000451
2: 0.0119557, 0.0124705, 0.0119475, 0.0124715, -0.0001727, 0.0001839
3: -0.0023154, -0.0017829, -0.0023238, -0.0017819, -0.0001786, 0.0001902
4: -0.0021069, -0.0015304, -0.0021080, -0.0015214, -0.0002059, 0.0001933
5: 0.0055617, 0.0061072, 0.0055531, 0.0061082, -0.0001830, 0.0001949
6: -0.0002333, 0.0019311, -0.0002674, 0.0019353, -0.0007259, 0.0007731
7: -0.0051868, -0.0022390, -0.0051924, -0.0021925, -0.0010529, 0.0009886
8: 0.9855602, 0.9876367, 0.9855563, 0.9876693, -0.0007417, 0.0006964
9: -0.0046647, -0.0027798, -0.0046944, -0.0027762, -0.0006321, 0.0006733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0034156, 0.0043632, -0.0004404, 0.0003715
1: 0.0018078, 0.0019517, 0.0018158, 0.0019527, -0.0000636, 0.0000537
2: 0.0119511, 0.0125021, 0.0119475, 0.0124715, -0.0002054, 0.0002435
3: -0.0023200, -0.0017502, -0.0023238, -0.0017819, -0.0002124, 0.0002518
4: -0.0021422, -0.0015254, -0.0021080, -0.0015214, -0.0002726, 0.0002299
5: 0.0055569, 0.0061406, 0.0055531, 0.0061082, -0.0002176, 0.0002580
6: -0.0002522, 0.0020639, -0.0002674, 0.0019353, -0.0008634, 0.0010237
7: -0.0053675, -0.0022132, -0.0051924, -0.0021925, -0.0013942, 0.0011759
8: 0.9854329, 0.9876548, 0.9855563, 0.9876693, -0.0009821, 0.0008283
9: -0.0046812, -0.0026642, -0.0046944, -0.0027762, -0.0007519, 0.0008915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033786, 0.0042932, -0.0004293, 0.0004531
1: 0.0018123, 0.0019448, 0.0018104, 0.0019425, -0.0000620, 0.0000655
2: 0.0119778, 0.0124845, 0.0119862, 0.0124919, -0.0002505, 0.0002374
3: -0.0022924, -0.0017684, -0.0022837, -0.0017607, -0.0002591, 0.0002455
4: -0.0021226, -0.0015553, -0.0021308, -0.0015647, -0.0002658, 0.0002805
5: 0.0055851, 0.0061220, 0.0055941, 0.0061298, -0.0002654, 0.0002515
6: -0.0001401, 0.0019901, -0.0001046, 0.0020211, -0.0010531, 0.0009979
7: -0.0052671, -0.0023659, -0.0053093, -0.0024142, -0.0013591, 0.0014342
8: 0.9855036, 0.9875473, 0.9854739, 0.9875132, -0.0009574, 0.0010103
9: -0.0045835, -0.0027284, -0.0045526, -0.0027014, -0.0009171, 0.0008690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005890
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005890
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033786, 0.0042932, -0.0005007, 0.0004789
1: 0.0018038, 0.0019463, 0.0018104, 0.0019425, -0.0000723, 0.0000692
2: 0.0119718, 0.0125172, 0.0119862, 0.0124919, -0.0002648, 0.0002768
3: -0.0022986, -0.0017346, -0.0022837, -0.0017607, -0.0002739, 0.0002863
4: -0.0021592, -0.0015485, -0.0021308, -0.0015647, -0.0003099, 0.0002965
5: 0.0055788, 0.0061567, 0.0055941, 0.0061298, -0.0002806, 0.0002933
6: -0.0001653, 0.0021275, -0.0001046, 0.0020211, -0.0011131, 0.0011638
7: -0.0054542, -0.0023316, -0.0053093, -0.0024142, -0.0015850, 0.0015160
8: 0.9853718, 0.9875715, 0.9854739, 0.9875132, -0.0011165, 0.0010679
9: -0.0046055, -0.0026088, -0.0045526, -0.0027014, -0.0009694, 0.0010135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005890
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005890
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0034168, 0.0043418, -0.0004254, 0.0003052
1: 0.0018123, 0.0019448, 0.0018159, 0.0019496, -0.0000615, 0.0000441
2: 0.0119778, 0.0124845, 0.0119594, 0.0124708, -0.0001687, 0.0002352
3: -0.0022924, -0.0017684, -0.0023115, -0.0017825, -0.0001745, 0.0002433
4: -0.0021226, -0.0015553, -0.0021072, -0.0015346, -0.0002633, 0.0001889
5: 0.0055851, 0.0061220, 0.0055656, 0.0061075, -0.0001788, 0.0002492
6: -0.0001401, 0.0019901, -0.0002176, 0.0019325, -0.0007094, 0.0009888
7: -0.0052671, -0.0023659, -0.0051886, -0.0022604, -0.0013466, 0.0009661
8: 0.9855036, 0.9875473, 0.9855589, 0.9876217, -0.0009486, 0.0006805
9: -0.0045835, -0.0027284, -0.0046510, -0.0027786, -0.0006177, 0.0008611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0034168, 0.0043418, -0.0005000, 0.0003640
1: 0.0018038, 0.0019463, 0.0018159, 0.0019496, -0.0000722, 0.0000526
2: 0.0119718, 0.0125172, 0.0119594, 0.0124708, -0.0002012, 0.0002764
3: -0.0022986, -0.0017346, -0.0023115, -0.0017825, -0.0002081, 0.0002859
4: -0.0021592, -0.0015485, -0.0021072, -0.0015346, -0.0003095, 0.0002253
5: 0.0055788, 0.0061567, 0.0055656, 0.0061075, -0.0002132, 0.0002929
6: -0.0001653, 0.0021275, -0.0002176, 0.0019325, -0.0008459, 0.0011620
7: -0.0054542, -0.0023316, -0.0051886, -0.0022604, -0.0015826, 0.0011521
8: 0.9853718, 0.9875715, 0.9855589, 0.9876217, -0.0011148, 0.0008115
9: -0.0046055, -0.0026088, -0.0046510, -0.0027786, -0.0007367, 0.0010119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033209, 0.0043199, -0.0004404, 0.0005692
1: 0.0018160, 0.0019505, 0.0018021, 0.0019464, -0.0000636, 0.0000822
2: 0.0119557, 0.0124705, 0.0119715, 0.0125238, -0.0003147, 0.0002435
3: -0.0023154, -0.0017829, -0.0022990, -0.0017278, -0.0003255, 0.0002518
4: -0.0021069, -0.0015304, -0.0021666, -0.0015482, -0.0002726, 0.0003523
5: 0.0055617, 0.0061072, 0.0055784, 0.0061636, -0.0003334, 0.0002580
6: -0.0002333, 0.0019311, -0.0001667, 0.0021552, -0.0013229, 0.0010237
7: -0.0051868, -0.0022390, -0.0054919, -0.0023297, -0.0013942, 0.0018017
8: 0.9855602, 0.9876367, 0.9853452, 0.9875728, -0.0009821, 0.0012691
9: -0.0046647, -0.0027798, -0.0046067, -0.0025847, -0.0011520, 0.0008915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006153
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006153
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033209, 0.0043199, -0.0004016, 0.0004784
1: 0.0018078, 0.0019517, 0.0018021, 0.0019464, -0.0000580, 0.0000691
2: 0.0119511, 0.0125021, 0.0119715, 0.0125238, -0.0002645, 0.0002220
3: -0.0023200, -0.0017502, -0.0022990, -0.0017278, -0.0002736, 0.0002296
4: -0.0021422, -0.0015254, -0.0021666, -0.0015482, -0.0002486, 0.0002962
5: 0.0055569, 0.0061406, 0.0055784, 0.0061636, -0.0002803, 0.0002353
6: -0.0002522, 0.0020639, -0.0001667, 0.0021552, -0.0011120, 0.0009334
7: -0.0053675, -0.0022132, -0.0054919, -0.0023297, -0.0012713, 0.0015145
8: 0.9854329, 0.9876548, 0.9853452, 0.9875728, -0.0008955, 0.0010668
9: -0.0046812, -0.0026642, -0.0046067, -0.0025847, -0.0009684, 0.0008129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005946
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005946
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033584, 0.0043726, -0.0003883, 0.0004202
1: 0.0018160, 0.0019505, 0.0018075, 0.0019540, -0.0000561, 0.0000607
2: 0.0119557, 0.0124705, 0.0119423, 0.0125031, -0.0002323, 0.0002147
3: -0.0023154, -0.0017829, -0.0023291, -0.0017492, -0.0002403, 0.0002220
4: -0.0021069, -0.0015304, -0.0021434, -0.0015155, -0.0002404, 0.0002601
5: 0.0055617, 0.0061072, 0.0055476, 0.0061417, -0.0002462, 0.0002275
6: -0.0002333, 0.0019311, -0.0002892, 0.0020682, -0.0009767, 0.0009025
7: -0.0051868, -0.0022390, -0.0053734, -0.0021628, -0.0012292, 0.0013302
8: 0.9855602, 0.9876367, 0.9854288, 0.9876904, -0.0008658, 0.0009370
9: -0.0046647, -0.0027798, -0.0047134, -0.0026605, -0.0008506, 0.0007860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006133
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006133
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033584, 0.0043726, -0.0003512, 0.0003316
1: 0.0018078, 0.0019517, 0.0018075, 0.0019540, -0.0000507, 0.0000479
2: 0.0119511, 0.0125021, 0.0119423, 0.0125031, -0.0001833, 0.0001942
3: -0.0023200, -0.0017502, -0.0023291, -0.0017492, -0.0001896, 0.0002008
4: -0.0021422, -0.0015254, -0.0021434, -0.0015155, -0.0002174, 0.0002053
5: 0.0055569, 0.0061406, 0.0055476, 0.0061417, -0.0001942, 0.0002057
6: -0.0002522, 0.0020639, -0.0002892, 0.0020682, -0.0007707, 0.0008163
7: -0.0053675, -0.0022132, -0.0053734, -0.0021628, -0.0011118, 0.0010496
8: 0.9854329, 0.9876548, 0.9854288, 0.9876904, -0.0007832, 0.0007394
9: -0.0046812, -0.0026642, -0.0047134, -0.0026605, -0.0006711, 0.0007109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005901
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005901
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033221, 0.0043000, -0.0004685, 0.0005616
1: 0.0018123, 0.0019448, 0.0018022, 0.0019435, -0.0000677, 0.0000811
2: 0.0119778, 0.0124845, 0.0119825, 0.0125232, -0.0003105, 0.0002590
3: -0.0022924, -0.0017684, -0.0022876, -0.0017284, -0.0003211, 0.0002679
4: -0.0021226, -0.0015553, -0.0021659, -0.0015605, -0.0002900, 0.0003476
5: 0.0055851, 0.0061220, 0.0055901, 0.0061630, -0.0003290, 0.0002744
6: -0.0001401, 0.0019901, -0.0001205, 0.0021525, -0.0013053, 0.0010889
7: -0.0052671, -0.0023659, -0.0054883, -0.0023927, -0.0014830, 0.0017777
8: 0.9855036, 0.9875473, 0.9853477, 0.9875284, -0.0010446, 0.0012522
9: -0.0045835, -0.0027284, -0.0045664, -0.0025870, -0.0011367, 0.0009482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006100
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006100
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033221, 0.0043000, -0.0004435, 0.0004701
1: 0.0018038, 0.0019463, 0.0018022, 0.0019435, -0.0000641, 0.0000679
2: 0.0119718, 0.0125172, 0.0119825, 0.0125232, -0.0002599, 0.0002452
3: -0.0022986, -0.0017346, -0.0022876, -0.0017284, -0.0002688, 0.0002536
4: -0.0021592, -0.0015485, -0.0021659, -0.0015605, -0.0002745, 0.0002910
5: 0.0055788, 0.0061567, 0.0055901, 0.0061630, -0.0002754, 0.0002598
6: -0.0001653, 0.0021275, -0.0001205, 0.0021525, -0.0010927, 0.0010308
7: -0.0054542, -0.0023316, -0.0054883, -0.0023927, -0.0014039, 0.0014881
8: 0.9853718, 0.9875715, 0.9853477, 0.9875284, -0.0009889, 0.0010483
9: -0.0046055, -0.0026088, -0.0045664, -0.0025870, -0.0009515, 0.0008977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005911
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005911
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033596, 0.0043522, -0.0004654, 0.0004131
1: 0.0018123, 0.0019448, 0.0018077, 0.0019511, -0.0000672, 0.0000597
2: 0.0119778, 0.0124845, 0.0119536, 0.0125024, -0.0002284, 0.0002573
3: -0.0022924, -0.0017684, -0.0023175, -0.0017499, -0.0002362, 0.0002661
4: -0.0021226, -0.0015553, -0.0021426, -0.0015282, -0.0002881, 0.0002557
5: 0.0055851, 0.0061220, 0.0055595, 0.0061410, -0.0002420, 0.0002726
6: -0.0001401, 0.0019901, -0.0002418, 0.0020653, -0.0009601, 0.0010817
7: -0.0052671, -0.0023659, -0.0053694, -0.0022274, -0.0014732, 0.0013075
8: 0.9855036, 0.9875473, 0.9854315, 0.9876449, -0.0010377, 0.0009211
9: -0.0045835, -0.0027284, -0.0046721, -0.0026630, -0.0008361, 0.0009420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005888, upper bound: 0.0006069
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005888, upper bound: 0.0006069
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033596, 0.0043522, -0.0004409, 0.0003248
1: 0.0018038, 0.0019463, 0.0018077, 0.0019511, -0.0000637, 0.0000469
2: 0.0119718, 0.0125172, 0.0119536, 0.0125024, -0.0001796, 0.0002438
3: -0.0022986, -0.0017346, -0.0023175, -0.0017499, -0.0001857, 0.0002521
4: -0.0021592, -0.0015485, -0.0021426, -0.0015282, -0.0002729, 0.0002011
5: 0.0055788, 0.0061567, 0.0055595, 0.0061410, -0.0001903, 0.0002583
6: -0.0001653, 0.0021275, -0.0002418, 0.0020653, -0.0007550, 0.0010248
7: -0.0054542, -0.0023316, -0.0053694, -0.0022274, -0.0013957, 0.0010283
8: 0.9853718, 0.9875715, 0.9854315, 0.9876449, -0.0009832, 0.0007243
9: -0.0046055, -0.0026088, -0.0046721, -0.0026630, -0.0006575, 0.0008925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005888, upper bound: 0.0005830
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005888, upper bound: 0.0005830
time: 0.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.65 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005933
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005933
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005933
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005933
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005962
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005962
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005962
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005962
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005896
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005896
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005896
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005891, upper bound: 0.0005896
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005888
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005889
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005889
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005889
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006159
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006160
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005950
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005950
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0006234
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0006234
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005986
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005986
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006106
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0006106
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005914
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005890, upper bound: 0.0005914
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0006158
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0006158
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005914
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005884, upper bound: 0.0005914
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005927
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005927
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005926
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005926
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005880
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005890
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005890
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005890
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005890
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005808
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006153
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006153
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005946
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005946
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006133
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006133
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005901
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005901
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006100
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0006100
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005911
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005889, upper bound: 0.0005911
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005888, upper bound: 0.0006069
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005888, upper bound: 0.0006069
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005888, upper bound: 0.0005830
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 8, lower bound: -0.0005888, upper bound: 0.0005830

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033794, 0.0042970, -0.0003060, 0.0003060
1: 0.0018105, 0.0019431, 0.0018105, 0.0019431, -0.0000442, 0.0000442
2: 0.0119841, 0.0124915, 0.0119841, 0.0124915, -0.0001692, 0.0001692
3: -0.0022859, -0.0017612, -0.0022859, -0.0017612, -0.0001750, 0.0001750
4: -0.0021303, -0.0015624, -0.0021303, -0.0015624, -0.0001894, 0.0001894
5: 0.0055919, 0.0061294, 0.0055919, 0.0061294, -0.0001793, 0.0001793
6: -0.0001135, 0.0020192, -0.0001135, 0.0020192, -0.0007112, 0.0007112
7: -0.0053067, -0.0024022, -0.0053067, -0.0024022, -0.0009686, 0.0009686
8: 0.9854757, 0.9875218, 0.9854757, 0.9875218, -0.0006823, 0.0006823
9: -0.0045603, -0.0027031, -0.0045603, -0.0027031, -0.0006194, 0.0006194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005751
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005667
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033497, 0.0042613, -0.0003317, 0.0003938
1: 0.0018105, 0.0019431, 0.0018062, 0.0019379, -0.0000479, 0.0000569
2: 0.0119841, 0.0124915, 0.0120039, 0.0125079, -0.0002177, 0.0001834
3: -0.0022859, -0.0017612, -0.0022654, -0.0017442, -0.0002252, 0.0001897
4: -0.0021303, -0.0015624, -0.0021488, -0.0015845, -0.0002053, 0.0002438
5: 0.0055919, 0.0061294, 0.0056128, 0.0061468, -0.0002307, 0.0001943
6: -0.0001135, 0.0020192, -0.0000304, 0.0020885, -0.0009154, 0.0007709
7: -0.0053067, -0.0024022, -0.0054010, -0.0025153, -0.0010499, 0.0012467
8: 0.9854757, 0.9875218, 0.9854093, 0.9874420, -0.0007396, 0.0008782
9: -0.0045603, -0.0027031, -0.0044880, -0.0026428, -0.0007972, 0.0006714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005751
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005667
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033794, 0.0042970, -0.0004145, 0.0003653
1: 0.0018024, 0.0019441, 0.0018105, 0.0019431, -0.0000599, 0.0000528
2: 0.0119803, 0.0125228, 0.0119841, 0.0124915, -0.0002019, 0.0002292
3: -0.0022898, -0.0017288, -0.0022859, -0.0017612, -0.0002089, 0.0002370
4: -0.0021654, -0.0015581, -0.0021303, -0.0015624, -0.0002566, 0.0002261
5: 0.0055878, 0.0061625, 0.0055919, 0.0061294, -0.0002140, 0.0002428
6: -0.0001295, 0.0021509, -0.0001135, 0.0020192, -0.0008490, 0.0009635
7: -0.0054860, -0.0023803, -0.0053067, -0.0024022, -0.0013122, 0.0011563
8: 0.9853494, 0.9875371, 0.9854757, 0.9875218, -0.0009244, 0.0008145
9: -0.0045743, -0.0025884, -0.0045603, -0.0027031, -0.0007393, 0.0008391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005821, upper bound: 0.0005746
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005825, upper bound: 0.0005666
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033497, 0.0042613, -0.0004402, 0.0004531
1: 0.0018024, 0.0019441, 0.0018062, 0.0019379, -0.0000636, 0.0000655
2: 0.0119803, 0.0125228, 0.0120039, 0.0125079, -0.0002505, 0.0002434
3: -0.0022898, -0.0017288, -0.0022654, -0.0017442, -0.0002591, 0.0002517
4: -0.0021654, -0.0015581, -0.0021488, -0.0015845, -0.0002725, 0.0002805
5: 0.0055878, 0.0061625, 0.0056128, 0.0061468, -0.0002654, 0.0002579
6: -0.0001295, 0.0021509, -0.0000304, 0.0020885, -0.0010532, 0.0010232
7: -0.0054860, -0.0023803, -0.0054010, -0.0025153, -0.0013935, 0.0014343
8: 0.9853494, 0.9875371, 0.9854093, 0.9874420, -0.0009816, 0.0010104
9: -0.0045743, -0.0025884, -0.0044880, -0.0026428, -0.0009171, 0.0008911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005821, upper bound: 0.0005746
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005825, upper bound: 0.0005666
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0034173, 0.0043486, -0.0004589, 0.0003626
1: 0.0018105, 0.0019431, 0.0018160, 0.0019505, -0.0000663, 0.0000524
2: 0.0119841, 0.0124915, 0.0119557, 0.0124705, -0.0002005, 0.0002537
3: -0.0022859, -0.0017612, -0.0023154, -0.0017829, -0.0002073, 0.0002624
4: -0.0021303, -0.0015624, -0.0021069, -0.0015304, -0.0002841, 0.0002245
5: 0.0055919, 0.0061294, 0.0055617, 0.0061072, -0.0002124, 0.0002689
6: -0.0001135, 0.0020192, -0.0002333, 0.0019311, -0.0008428, 0.0010667
7: -0.0053067, -0.0024022, -0.0051868, -0.0022390, -0.0014528, 0.0011478
8: 0.9854757, 0.9875218, 0.9855602, 0.9876367, -0.0010234, 0.0008085
9: -0.0045603, -0.0027031, -0.0046647, -0.0027798, -0.0007339, 0.0009289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005693
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005594
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033920, 0.0043085, -0.0004327, 0.0004008
1: 0.0018105, 0.0019431, 0.0018123, 0.0019448, -0.0000625, 0.0000579
2: 0.0119841, 0.0124915, 0.0119778, 0.0124845, -0.0002216, 0.0002392
3: -0.0022859, -0.0017612, -0.0022924, -0.0017684, -0.0002292, 0.0002474
4: -0.0021303, -0.0015624, -0.0021226, -0.0015553, -0.0002678, 0.0002481
5: 0.0055919, 0.0061294, 0.0055851, 0.0061220, -0.0002348, 0.0002535
6: -0.0001135, 0.0020192, -0.0001401, 0.0019901, -0.0009316, 0.0010057
7: -0.0053067, -0.0024022, -0.0052671, -0.0023659, -0.0013697, 0.0012687
8: 0.9854757, 0.9875218, 0.9855036, 0.9875473, -0.0009648, 0.0008937
9: -0.0045603, -0.0027031, -0.0045835, -0.0027284, -0.0008113, 0.0008758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005693
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005594
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0034173, 0.0043486, -0.0005675, 0.0004219
1: 0.0018024, 0.0019441, 0.0018160, 0.0019505, -0.0000820, 0.0000609
2: 0.0119803, 0.0125228, 0.0119557, 0.0124705, -0.0002332, 0.0003138
3: -0.0022898, -0.0017288, -0.0023154, -0.0017829, -0.0002412, 0.0003245
4: -0.0021654, -0.0015581, -0.0021069, -0.0015304, -0.0003513, 0.0002611
5: 0.0055878, 0.0061625, 0.0055617, 0.0061072, -0.0002471, 0.0003324
6: -0.0001295, 0.0021509, -0.0002333, 0.0019311, -0.0009806, 0.0013190
7: -0.0054860, -0.0023803, -0.0051868, -0.0022390, -0.0017964, 0.0013354
8: 0.9853494, 0.9875371, 0.9855602, 0.9876367, -0.0012654, 0.0009407
9: -0.0045743, -0.0025884, -0.0046647, -0.0027798, -0.0008539, 0.0011487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005763, upper bound: 0.0005686
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005594
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033920, 0.0043085, -0.0005412, 0.0004601
1: 0.0018024, 0.0019441, 0.0018123, 0.0019448, -0.0000782, 0.0000665
2: 0.0119803, 0.0125228, 0.0119778, 0.0124845, -0.0002544, 0.0002992
3: -0.0022898, -0.0017288, -0.0022924, -0.0017684, -0.0002631, 0.0003095
4: -0.0021654, -0.0015581, -0.0021226, -0.0015553, -0.0003350, 0.0002848
5: 0.0055878, 0.0061625, 0.0055851, 0.0061220, -0.0002695, 0.0003171
6: -0.0001295, 0.0021509, -0.0001401, 0.0019901, -0.0010694, 0.0012580
7: -0.0054860, -0.0023803, -0.0052671, -0.0023659, -0.0017133, 0.0014564
8: 0.9853494, 0.9875371, 0.9855036, 0.9875473, -0.0012069, 0.0010259
9: -0.0045743, -0.0025884, -0.0045835, -0.0027284, -0.0009312, 0.0010955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005763, upper bound: 0.0005686
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005594
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033794, 0.0042970, -0.0003938, 0.0003317
1: 0.0018062, 0.0019379, 0.0018105, 0.0019431, -0.0000569, 0.0000479
2: 0.0120039, 0.0125079, 0.0119841, 0.0124915, -0.0001834, 0.0002177
3: -0.0022654, -0.0017442, -0.0022859, -0.0017612, -0.0001897, 0.0002252
4: -0.0021488, -0.0015845, -0.0021303, -0.0015624, -0.0002438, 0.0002053
5: 0.0056128, 0.0061468, 0.0055919, 0.0061294, -0.0001943, 0.0002307
6: -0.0000304, 0.0020885, -0.0001135, 0.0020192, -0.0007709, 0.0009154
7: -0.0054010, -0.0025153, -0.0053067, -0.0024022, -0.0012467, 0.0010499
8: 0.9854093, 0.9874420, 0.9854757, 0.9875218, -0.0008782, 0.0007396
9: -0.0044880, -0.0026428, -0.0045603, -0.0027031, -0.0006714, 0.0007972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005699
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005627
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033497, 0.0042613, -0.0003092, 0.0003092
1: 0.0018062, 0.0019379, 0.0018062, 0.0019379, -0.0000447, 0.0000447
2: 0.0120039, 0.0125079, 0.0120039, 0.0125079, -0.0001709, 0.0001709
3: -0.0022654, -0.0017442, -0.0022654, -0.0017442, -0.0001768, 0.0001768
4: -0.0021488, -0.0015845, -0.0021488, -0.0015845, -0.0001914, 0.0001914
5: 0.0056128, 0.0061468, 0.0056128, 0.0061468, -0.0001811, 0.0001811
6: -0.0000304, 0.0020885, -0.0000304, 0.0020885, -0.0007186, 0.0007186
7: -0.0054010, -0.0025153, -0.0054010, -0.0025153, -0.0009787, 0.0009787
8: 0.9854093, 0.9874420, 0.9854093, 0.9874420, -0.0006894, 0.0006894
9: -0.0044880, -0.0026428, -0.0044880, -0.0026428, -0.0006258, 0.0006258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005699
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005627
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033794, 0.0042970, -0.0004696, 0.0003552
1: 0.0017983, 0.0019389, 0.0018105, 0.0019431, -0.0000678, 0.0000513
2: 0.0120001, 0.0125384, 0.0119841, 0.0124915, -0.0001964, 0.0002596
3: -0.0022694, -0.0017127, -0.0022859, -0.0017612, -0.0002031, 0.0002685
4: -0.0021829, -0.0015802, -0.0021303, -0.0015624, -0.0002907, 0.0002199
5: 0.0056088, 0.0061791, 0.0055919, 0.0061294, -0.0002081, 0.0002751
6: -0.0000464, 0.0022166, -0.0001135, 0.0020192, -0.0008257, 0.0010914
7: -0.0055755, -0.0024935, -0.0053067, -0.0024022, -0.0014864, 0.0011245
8: 0.9852863, 0.9874574, 0.9854757, 0.9875218, -0.0010471, 0.0007921
9: -0.0045019, -0.0025312, -0.0045603, -0.0027031, -0.0007191, 0.0009505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005821, upper bound: 0.0005699
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005825, upper bound: 0.0005627
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033497, 0.0042613, -0.0004165, 0.0003674
1: 0.0017983, 0.0019389, 0.0018062, 0.0019379, -0.0000602, 0.0000531
2: 0.0120001, 0.0125384, 0.0120039, 0.0125079, -0.0002031, 0.0002303
3: -0.0022694, -0.0017127, -0.0022654, -0.0017442, -0.0002101, 0.0002382
4: -0.0021829, -0.0015802, -0.0021488, -0.0015845, -0.0002578, 0.0002274
5: 0.0056088, 0.0061791, 0.0056128, 0.0061468, -0.0002152, 0.0002440
6: -0.0000464, 0.0022166, -0.0000304, 0.0020885, -0.0008539, 0.0009681
7: -0.0055755, -0.0024935, -0.0054010, -0.0025153, -0.0013185, 0.0011630
8: 0.9852863, 0.9874574, 0.9854093, 0.9874420, -0.0009287, 0.0008192
9: -0.0045019, -0.0025312, -0.0044880, -0.0026428, -0.0007436, 0.0008431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005821, upper bound: 0.0005699
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005825, upper bound: 0.0005627
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0034173, 0.0043486, -0.0005468, 0.0003883
1: 0.0018062, 0.0019379, 0.0018160, 0.0019505, -0.0000790, 0.0000561
2: 0.0120039, 0.0125079, 0.0119557, 0.0124705, -0.0002147, 0.0003023
3: -0.0022654, -0.0017442, -0.0023154, -0.0017829, -0.0002220, 0.0003127
4: -0.0021488, -0.0015845, -0.0021069, -0.0015304, -0.0003385, 0.0002404
5: 0.0056128, 0.0061468, 0.0055617, 0.0061072, -0.0002275, 0.0003203
6: -0.0000304, 0.0020885, -0.0002333, 0.0019311, -0.0009025, 0.0012709
7: -0.0054010, -0.0025153, -0.0051868, -0.0022390, -0.0017308, 0.0012291
8: 0.9854093, 0.9874420, 0.9855602, 0.9876367, -0.0012192, 0.0008658
9: -0.0044880, -0.0026428, -0.0046647, -0.0027798, -0.0007859, 0.0011067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005597
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005492
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033920, 0.0043085, -0.0004591, 0.0003612
1: 0.0018062, 0.0019379, 0.0018123, 0.0019448, -0.0000663, 0.0000522
2: 0.0120039, 0.0125079, 0.0119778, 0.0124845, -0.0001997, 0.0002538
3: -0.0022654, -0.0017442, -0.0022924, -0.0017684, -0.0002065, 0.0002625
4: -0.0021488, -0.0015845, -0.0021226, -0.0015553, -0.0002842, 0.0002236
5: 0.0056128, 0.0061468, 0.0055851, 0.0061220, -0.0002116, 0.0002690
6: -0.0000304, 0.0020885, -0.0001401, 0.0019901, -0.0008395, 0.0010672
7: -0.0054010, -0.0025153, -0.0052671, -0.0023659, -0.0014534, 0.0011433
8: 0.9854093, 0.9874420, 0.9855036, 0.9875473, -0.0010238, 0.0008054
9: -0.0044880, -0.0026428, -0.0045835, -0.0027284, -0.0007311, 0.0009293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005597
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005492
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0034173, 0.0043486, -0.0006225, 0.0004119
1: 0.0017983, 0.0019389, 0.0018160, 0.0019505, -0.0000899, 0.0000595
2: 0.0120001, 0.0125384, 0.0119557, 0.0124705, -0.0002277, 0.0003442
3: -0.0022694, -0.0017127, -0.0023154, -0.0017829, -0.0002355, 0.0003560
4: -0.0021829, -0.0015802, -0.0021069, -0.0015304, -0.0003854, 0.0002549
5: 0.0056088, 0.0061791, 0.0055617, 0.0061072, -0.0002413, 0.0003647
6: -0.0000464, 0.0022166, -0.0002333, 0.0019311, -0.0009573, 0.0014469
7: -0.0055755, -0.0024935, -0.0051868, -0.0022390, -0.0019706, 0.0013037
8: 0.9852863, 0.9874574, 0.9855602, 0.9876367, -0.0013881, 0.0009184
9: -0.0045019, -0.0025312, -0.0046647, -0.0027798, -0.0008336, 0.0012601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005763, upper bound: 0.0005596
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005492
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033920, 0.0043085, -0.0005665, 0.0004194
1: 0.0017983, 0.0019389, 0.0018123, 0.0019448, -0.0000818, 0.0000606
2: 0.0120001, 0.0125384, 0.0119778, 0.0124845, -0.0002319, 0.0003132
3: -0.0022694, -0.0017127, -0.0022924, -0.0017684, -0.0002398, 0.0003239
4: -0.0021829, -0.0015802, -0.0021226, -0.0015553, -0.0003507, 0.0002596
5: 0.0056088, 0.0061791, 0.0055851, 0.0061220, -0.0002457, 0.0003318
6: -0.0000464, 0.0022166, -0.0001401, 0.0019901, -0.0009748, 0.0013166
7: -0.0055755, -0.0024935, -0.0052671, -0.0023659, -0.0017932, 0.0013276
8: 0.9852863, 0.9874574, 0.9855036, 0.9875473, -0.0012631, 0.0009352
9: -0.0045019, -0.0025312, -0.0045835, -0.0027284, -0.0008489, 0.0011466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005763, upper bound: 0.0005596
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005492
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033228, 0.0043039, -0.0003653, 0.0004145
1: 0.0018105, 0.0019431, 0.0018024, 0.0019441, -0.0000528, 0.0000599
2: 0.0119841, 0.0124915, 0.0119803, 0.0125228, -0.0002292, 0.0002019
3: -0.0022859, -0.0017612, -0.0022898, -0.0017288, -0.0002370, 0.0002089
4: -0.0021303, -0.0015624, -0.0021654, -0.0015581, -0.0002261, 0.0002566
5: 0.0055919, 0.0061294, 0.0055878, 0.0061625, -0.0002428, 0.0002140
6: -0.0001135, 0.0020192, -0.0001295, 0.0021509, -0.0009635, 0.0008490
7: -0.0053067, -0.0024022, -0.0054860, -0.0023803, -0.0011563, 0.0013122
8: 0.9854757, 0.9875218, 0.9853494, 0.9875371, -0.0008145, 0.0009244
9: -0.0045603, -0.0027031, -0.0045743, -0.0025884, -0.0008391, 0.0007393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005974
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005877
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0032945, 0.0042682, -0.0003552, 0.0004696
1: 0.0018105, 0.0019431, 0.0017983, 0.0019389, -0.0000513, 0.0000678
2: 0.0119841, 0.0124915, 0.0120001, 0.0125384, -0.0002596, 0.0001964
3: -0.0022859, -0.0017612, -0.0022694, -0.0017127, -0.0002685, 0.0002031
4: -0.0021303, -0.0015624, -0.0021829, -0.0015802, -0.0002199, 0.0002907
5: 0.0055919, 0.0061294, 0.0056088, 0.0061791, -0.0002751, 0.0002081
6: -0.0001135, 0.0020192, -0.0000464, 0.0022166, -0.0010914, 0.0008257
7: -0.0053067, -0.0024022, -0.0055755, -0.0024935, -0.0011245, 0.0014864
8: 0.9854757, 0.9875218, 0.9852863, 0.9874574, -0.0007921, 0.0010471
9: -0.0045603, -0.0027031, -0.0045019, -0.0025312, -0.0009505, 0.0007191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005974
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005876
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033228, 0.0043039, -0.0003239, 0.0003239
1: 0.0018024, 0.0019441, 0.0018024, 0.0019441, -0.0000468, 0.0000468
2: 0.0119803, 0.0125228, 0.0119803, 0.0125228, -0.0001791, 0.0001791
3: -0.0022898, -0.0017288, -0.0022898, -0.0017288, -0.0001852, 0.0001852
4: -0.0021654, -0.0015581, -0.0021654, -0.0015581, -0.0002005, 0.0002005
5: 0.0055878, 0.0061625, 0.0055878, 0.0061625, -0.0001898, 0.0001898
6: -0.0001295, 0.0021509, -0.0001295, 0.0021509, -0.0007529, 0.0007529
7: -0.0054860, -0.0023803, -0.0054860, -0.0023803, -0.0010254, 0.0010254
8: 0.9853494, 0.9875371, 0.9853494, 0.9875371, -0.0007223, 0.0007223
9: -0.0045743, -0.0025884, -0.0045743, -0.0025884, -0.0006557, 0.0006557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005881, upper bound: 0.0005785
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005892, upper bound: 0.0005725
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0032945, 0.0042682, -0.0003486, 0.0004086
1: 0.0018024, 0.0019441, 0.0017983, 0.0019389, -0.0000504, 0.0000590
2: 0.0119803, 0.0125228, 0.0120001, 0.0125384, -0.0002259, 0.0001927
3: -0.0022898, -0.0017288, -0.0022694, -0.0017127, -0.0002336, 0.0001993
4: -0.0021654, -0.0015581, -0.0021829, -0.0015802, -0.0002158, 0.0002529
5: 0.0055878, 0.0061625, 0.0056088, 0.0061791, -0.0002393, 0.0002042
6: -0.0001295, 0.0021509, -0.0000464, 0.0022166, -0.0009496, 0.0008102
7: -0.0054860, -0.0023803, -0.0055755, -0.0024935, -0.0011035, 0.0012933
8: 0.9853494, 0.9875371, 0.9852863, 0.9874574, -0.0007773, 0.0009110
9: -0.0045743, -0.0025884, -0.0045019, -0.0025312, -0.0008270, 0.0007056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005881, upper bound: 0.0005786
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005892, upper bound: 0.0005725
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033602, 0.0043567, -0.0004852, 0.0004368
1: 0.0018105, 0.0019431, 0.0018078, 0.0019517, -0.0000701, 0.0000631
2: 0.0119841, 0.0124915, 0.0119511, 0.0125021, -0.0002415, 0.0002683
3: -0.0022859, -0.0017612, -0.0023200, -0.0017502, -0.0002498, 0.0002775
4: -0.0021303, -0.0015624, -0.0021422, -0.0015254, -0.0003004, 0.0002704
5: 0.0055919, 0.0061294, 0.0055569, 0.0061406, -0.0002559, 0.0002843
6: -0.0001135, 0.0020192, -0.0002522, 0.0020639, -0.0010153, 0.0011278
7: -0.0053067, -0.0024022, -0.0053675, -0.0022132, -0.0015360, 0.0013828
8: 0.9854757, 0.9875218, 0.9854329, 0.9876548, -0.0010820, 0.0009741
9: -0.0045603, -0.0027031, -0.0046812, -0.0026642, -0.0008842, 0.0009822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005953
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005867
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042970, 0.0033329, 0.0043193, -0.0004538, 0.0004722
1: 0.0018105, 0.0019431, 0.0018038, 0.0019463, -0.0000656, 0.0000682
2: 0.0119841, 0.0124915, 0.0119718, 0.0125172, -0.0002610, 0.0002509
3: -0.0022859, -0.0017612, -0.0022986, -0.0017346, -0.0002700, 0.0002595
4: -0.0021303, -0.0015624, -0.0021592, -0.0015485, -0.0002809, 0.0002923
5: 0.0055919, 0.0061294, 0.0055788, 0.0061567, -0.0002766, 0.0002658
6: -0.0001135, 0.0020192, -0.0001653, 0.0021275, -0.0010974, 0.0010547
7: -0.0053067, -0.0024022, -0.0054542, -0.0023316, -0.0014364, 0.0014946
8: 0.9854757, 0.9875218, 0.9853718, 0.9875715, -0.0010118, 0.0010528
9: -0.0045603, -0.0027031, -0.0046055, -0.0026088, -0.0009557, 0.0009185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005953
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005867
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033602, 0.0043567, -0.0004768, 0.0003790
1: 0.0018024, 0.0019441, 0.0018078, 0.0019517, -0.0000689, 0.0000547
2: 0.0119803, 0.0125228, 0.0119511, 0.0125021, -0.0002095, 0.0002636
3: -0.0022898, -0.0017288, -0.0023200, -0.0017502, -0.0002167, 0.0002726
4: -0.0021654, -0.0015581, -0.0021422, -0.0015254, -0.0002952, 0.0002346
5: 0.0055878, 0.0061625, 0.0055569, 0.0061406, -0.0002220, 0.0002793
6: -0.0001295, 0.0021509, -0.0002522, 0.0020639, -0.0008808, 0.0011082
7: -0.0054860, -0.0023803, -0.0053675, -0.0022132, -0.0015093, 0.0011996
8: 0.9853494, 0.9875371, 0.9854329, 0.9876548, -0.0010632, 0.0008450
9: -0.0045743, -0.0025884, -0.0046812, -0.0026642, -0.0007671, 0.0009651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005853, upper bound: 0.0005797
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005857, upper bound: 0.0005723
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0043039, 0.0033329, 0.0043193, -0.0004497, 0.0004159
1: 0.0018024, 0.0019441, 0.0018038, 0.0019463, -0.0000650, 0.0000601
2: 0.0119803, 0.0125228, 0.0119718, 0.0125172, -0.0002299, 0.0002486
3: -0.0022898, -0.0017288, -0.0022986, -0.0017346, -0.0002378, 0.0002571
4: -0.0021654, -0.0015581, -0.0021592, -0.0015485, -0.0002784, 0.0002574
5: 0.0055878, 0.0061625, 0.0055788, 0.0061567, -0.0002436, 0.0002634
6: -0.0001295, 0.0021509, -0.0001653, 0.0021275, -0.0009666, 0.0010452
7: -0.0054860, -0.0023803, -0.0054542, -0.0023316, -0.0014235, 0.0013164
8: 0.9853494, 0.9875371, 0.9853718, 0.9875715, -0.0010027, 0.0009273
9: -0.0045743, -0.0025884, -0.0046055, -0.0026088, -0.0008418, 0.0009102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005853, upper bound: 0.0005797
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005857, upper bound: 0.0005723
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033228, 0.0043039, -0.0004531, 0.0004402
1: 0.0018062, 0.0019379, 0.0018024, 0.0019441, -0.0000655, 0.0000636
2: 0.0120039, 0.0125079, 0.0119803, 0.0125228, -0.0002434, 0.0002505
3: -0.0022654, -0.0017442, -0.0022898, -0.0017288, -0.0002517, 0.0002591
4: -0.0021488, -0.0015845, -0.0021654, -0.0015581, -0.0002805, 0.0002725
5: 0.0056128, 0.0061468, 0.0055878, 0.0061625, -0.0002579, 0.0002654
6: -0.0000304, 0.0020885, -0.0001295, 0.0021509, -0.0010232, 0.0010532
7: -0.0054010, -0.0025153, -0.0054860, -0.0023803, -0.0014343, 0.0013935
8: 0.9854093, 0.9874420, 0.9853494, 0.9875371, -0.0010104, 0.0009816
9: -0.0044880, -0.0026428, -0.0045743, -0.0025884, -0.0008911, 0.0009171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005906
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005826
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0032945, 0.0042682, -0.0003674, 0.0004165
1: 0.0018062, 0.0019379, 0.0017983, 0.0019389, -0.0000531, 0.0000602
2: 0.0120039, 0.0125079, 0.0120001, 0.0125384, -0.0002303, 0.0002031
3: -0.0022654, -0.0017442, -0.0022694, -0.0017127, -0.0002382, 0.0002101
4: -0.0021488, -0.0015845, -0.0021829, -0.0015802, -0.0002274, 0.0002578
5: 0.0056128, 0.0061468, 0.0056088, 0.0061791, -0.0002440, 0.0002152
6: -0.0000304, 0.0020885, -0.0000464, 0.0022166, -0.0009681, 0.0008539
7: -0.0054010, -0.0025153, -0.0055755, -0.0024935, -0.0011630, 0.0013185
8: 0.9854093, 0.9874420, 0.9852863, 0.9874574, -0.0008192, 0.0009287
9: -0.0044880, -0.0026428, -0.0045019, -0.0025312, -0.0008431, 0.0007436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005906
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005826
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033228, 0.0043039, -0.0004086, 0.0003486
1: 0.0017983, 0.0019389, 0.0018024, 0.0019441, -0.0000590, 0.0000504
2: 0.0120001, 0.0125384, 0.0119803, 0.0125228, -0.0001927, 0.0002259
3: -0.0022694, -0.0017127, -0.0022898, -0.0017288, -0.0001993, 0.0002336
4: -0.0021829, -0.0015802, -0.0021654, -0.0015581, -0.0002529, 0.0002158
5: 0.0056088, 0.0061791, 0.0055878, 0.0061625, -0.0002042, 0.0002393
6: -0.0000464, 0.0022166, -0.0001295, 0.0021509, -0.0008102, 0.0009496
7: -0.0055755, -0.0024935, -0.0054860, -0.0023803, -0.0012933, 0.0011035
8: 0.9852863, 0.9874574, 0.9853494, 0.9875371, -0.0009110, 0.0007773
9: -0.0045019, -0.0025312, -0.0045743, -0.0025884, -0.0007056, 0.0008270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005882, upper bound: 0.0005742
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005689
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0032945, 0.0042682, -0.0003255, 0.0003255
1: 0.0017983, 0.0019389, 0.0017983, 0.0019389, -0.0000470, 0.0000470
2: 0.0120001, 0.0125384, 0.0120001, 0.0125384, -0.0001800, 0.0001800
3: -0.0022694, -0.0017127, -0.0022694, -0.0017127, -0.0001861, 0.0001861
4: -0.0021829, -0.0015802, -0.0021829, -0.0015802, -0.0002015, 0.0002015
5: 0.0056088, 0.0061791, 0.0056088, 0.0061791, -0.0001907, 0.0001907
6: -0.0000464, 0.0022166, -0.0000464, 0.0022166, -0.0007566, 0.0007566
7: -0.0055755, -0.0024935, -0.0055755, -0.0024935, -0.0010304, 0.0010304
8: 0.9852863, 0.9874574, 0.9852863, 0.9874574, -0.0007258, 0.0007258
9: -0.0045019, -0.0025312, -0.0045019, -0.0025312, -0.0006589, 0.0006589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005882, upper bound: 0.0005742
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005689
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033602, 0.0043567, -0.0005731, 0.0004625
1: 0.0018062, 0.0019379, 0.0018078, 0.0019517, -0.0000828, 0.0000668
2: 0.0120039, 0.0125079, 0.0119511, 0.0125021, -0.0002557, 0.0003168
3: -0.0022654, -0.0017442, -0.0023200, -0.0017502, -0.0002645, 0.0003277
4: -0.0021488, -0.0015845, -0.0021422, -0.0015254, -0.0003547, 0.0002863
5: 0.0056128, 0.0061468, 0.0055569, 0.0061406, -0.0002709, 0.0003357
6: -0.0000304, 0.0020885, -0.0002522, 0.0020639, -0.0010750, 0.0013320
7: -0.0054010, -0.0025153, -0.0053675, -0.0022132, -0.0018141, 0.0014641
8: 0.9854093, 0.9874420, 0.9854329, 0.9876548, -0.0012779, 0.0010314
9: -0.0044880, -0.0026428, -0.0046812, -0.0026642, -0.0009362, 0.0011600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005890
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005757
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033497, 0.0042613, 0.0033329, 0.0043193, -0.0004850, 0.0004364
1: 0.0018062, 0.0019379, 0.0018038, 0.0019463, -0.0000701, 0.0000630
2: 0.0120039, 0.0125079, 0.0119718, 0.0125172, -0.0002413, 0.0002681
3: -0.0022654, -0.0017442, -0.0022986, -0.0017346, -0.0002495, 0.0002773
4: -0.0021488, -0.0015845, -0.0021592, -0.0015485, -0.0003002, 0.0002701
5: 0.0056128, 0.0061468, 0.0055788, 0.0061567, -0.0002556, 0.0002841
6: -0.0000304, 0.0020885, -0.0001653, 0.0021275, -0.0010143, 0.0011272
7: -0.0054010, -0.0025153, -0.0054542, -0.0023316, -0.0015352, 0.0013814
8: 0.9854093, 0.9874420, 0.9853718, 0.9875715, -0.0010814, 0.0009731
9: -0.0044880, -0.0026428, -0.0046055, -0.0026088, -0.0008833, 0.0009816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005890
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005757
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033602, 0.0043567, -0.0005614, 0.0004036
1: 0.0017983, 0.0019389, 0.0018078, 0.0019517, -0.0000811, 0.0000583
2: 0.0120001, 0.0125384, 0.0119511, 0.0125021, -0.0002232, 0.0003104
3: -0.0022694, -0.0017127, -0.0023200, -0.0017502, -0.0002308, 0.0003210
4: -0.0021829, -0.0015802, -0.0021422, -0.0015254, -0.0003475, 0.0002498
5: 0.0056088, 0.0061791, 0.0055569, 0.0061406, -0.0002364, 0.0003289
6: -0.0000464, 0.0022166, -0.0002522, 0.0020639, -0.0009381, 0.0013049
7: -0.0055755, -0.0024935, -0.0053675, -0.0022132, -0.0017772, 0.0012776
8: 0.9852863, 0.9874574, 0.9854329, 0.9876548, -0.0012519, 0.0009000
9: -0.0045019, -0.0025312, -0.0046812, -0.0026642, -0.0008170, 0.0011364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005853, upper bound: 0.0005717
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005857, upper bound: 0.0005648
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032945, 0.0042682, 0.0033329, 0.0043193, -0.0004758, 0.0003764
1: 0.0017983, 0.0019389, 0.0018038, 0.0019463, -0.0000687, 0.0000544
2: 0.0120001, 0.0125384, 0.0119718, 0.0125172, -0.0002081, 0.0002630
3: -0.0022694, -0.0017127, -0.0022986, -0.0017346, -0.0002152, 0.0002720
4: -0.0021829, -0.0015802, -0.0021592, -0.0015485, -0.0002945, 0.0002330
5: 0.0056088, 0.0061791, 0.0055788, 0.0061567, -0.0002205, 0.0002787
6: -0.0000464, 0.0022166, -0.0001653, 0.0021275, -0.0008749, 0.0011058
7: -0.0055755, -0.0024935, -0.0054542, -0.0023316, -0.0015060, 0.0011916
8: 0.9852863, 0.9874574, 0.9853718, 0.9875715, -0.0010609, 0.0008394
9: -0.0045019, -0.0025312, -0.0046055, -0.0026088, -0.0007619, 0.0009630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005853, upper bound: 0.0005717
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005857, upper bound: 0.0005648
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033794, 0.0042970, -0.0003626, 0.0004589
1: 0.0018160, 0.0019505, 0.0018105, 0.0019431, -0.0000524, 0.0000663
2: 0.0119557, 0.0124705, 0.0119841, 0.0124915, -0.0002537, 0.0002005
3: -0.0023154, -0.0017829, -0.0022859, -0.0017612, -0.0002624, 0.0002073
4: -0.0021069, -0.0015304, -0.0021303, -0.0015624, -0.0002245, 0.0002841
5: 0.0055617, 0.0061072, 0.0055919, 0.0061294, -0.0002689, 0.0002124
6: -0.0002333, 0.0019311, -0.0001135, 0.0020192, -0.0010667, 0.0008428
7: -0.0051868, -0.0022390, -0.0053067, -0.0024022, -0.0011478, 0.0014528
8: 0.9855602, 0.9876367, 0.9854757, 0.9875218, -0.0008085, 0.0010234
9: -0.0046647, -0.0027798, -0.0045603, -0.0027031, -0.0009289, 0.0007339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005697
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005613
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033497, 0.0042613, -0.0003883, 0.0005468
1: 0.0018160, 0.0019505, 0.0018062, 0.0019379, -0.0000561, 0.0000790
2: 0.0119557, 0.0124705, 0.0120039, 0.0125079, -0.0003023, 0.0002147
3: -0.0023154, -0.0017829, -0.0022654, -0.0017442, -0.0003127, 0.0002220
4: -0.0021069, -0.0015304, -0.0021488, -0.0015845, -0.0002404, 0.0003385
5: 0.0055617, 0.0061072, 0.0056128, 0.0061468, -0.0003203, 0.0002275
6: -0.0002333, 0.0019311, -0.0000304, 0.0020885, -0.0012709, 0.0009025
7: -0.0051868, -0.0022390, -0.0054010, -0.0025153, -0.0012291, 0.0017308
8: 0.9855602, 0.9876367, 0.9854093, 0.9874420, -0.0008658, 0.0012192
9: -0.0046647, -0.0027798, -0.0044880, -0.0026428, -0.0011067, 0.0007859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005697
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005613
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033794, 0.0042970, -0.0004368, 0.0004852
1: 0.0018078, 0.0019517, 0.0018105, 0.0019431, -0.0000631, 0.0000701
2: 0.0119511, 0.0125021, 0.0119841, 0.0124915, -0.0002683, 0.0002415
3: -0.0023200, -0.0017502, -0.0022859, -0.0017612, -0.0002775, 0.0002498
4: -0.0021422, -0.0015254, -0.0021303, -0.0015624, -0.0002704, 0.0003004
5: 0.0055569, 0.0061406, 0.0055919, 0.0061294, -0.0002843, 0.0002559
6: -0.0002522, 0.0020639, -0.0001135, 0.0020192, -0.0011278, 0.0010153
7: -0.0053675, -0.0022132, -0.0053067, -0.0024022, -0.0013828, 0.0015360
8: 0.9854329, 0.9876548, 0.9854757, 0.9875218, -0.0009741, 0.0010820
9: -0.0046812, -0.0026642, -0.0045603, -0.0027031, -0.0009822, 0.0008842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005696
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005613
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033497, 0.0042613, -0.0004625, 0.0005731
1: 0.0018078, 0.0019517, 0.0018062, 0.0019379, -0.0000668, 0.0000828
2: 0.0119511, 0.0125021, 0.0120039, 0.0125079, -0.0003168, 0.0002557
3: -0.0023200, -0.0017502, -0.0022654, -0.0017442, -0.0003277, 0.0002645
4: -0.0021422, -0.0015254, -0.0021488, -0.0015845, -0.0002863, 0.0003547
5: 0.0055569, 0.0061406, 0.0056128, 0.0061468, -0.0003357, 0.0002709
6: -0.0002522, 0.0020639, -0.0000304, 0.0020885, -0.0013320, 0.0010750
7: -0.0053675, -0.0022132, -0.0054010, -0.0025153, -0.0014641, 0.0018141
8: 0.9854329, 0.9876548, 0.9854093, 0.9874420, -0.0010314, 0.0012779
9: -0.0046812, -0.0026642, -0.0044880, -0.0026428, -0.0011600, 0.0009362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005696
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005613
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0034173, 0.0043486, -0.0003107, 0.0003107
1: 0.0018160, 0.0019505, 0.0018160, 0.0019505, -0.0000449, 0.0000449
2: 0.0119557, 0.0124705, 0.0119557, 0.0124705, -0.0001718, 0.0001718
3: -0.0023154, -0.0017829, -0.0023154, -0.0017829, -0.0001777, 0.0001777
4: -0.0021069, -0.0015304, -0.0021069, -0.0015304, -0.0001923, 0.0001923
5: 0.0055617, 0.0061072, 0.0055617, 0.0061072, -0.0001820, 0.0001820
6: -0.0002333, 0.0019311, -0.0002333, 0.0019311, -0.0007222, 0.0007222
7: -0.0051868, -0.0022390, -0.0051868, -0.0022390, -0.0009835, 0.0009835
8: 0.9855602, 0.9876367, 0.9855602, 0.9876367, -0.0006928, 0.0006928
9: -0.0046647, -0.0027798, -0.0046647, -0.0027798, -0.0006289, 0.0006289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005647
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005554
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033920, 0.0043085, -0.0003344, 0.0003987
1: 0.0018160, 0.0019505, 0.0018123, 0.0019448, -0.0000483, 0.0000576
2: 0.0119557, 0.0124705, 0.0119778, 0.0124845, -0.0002204, 0.0001849
3: -0.0023154, -0.0017829, -0.0022924, -0.0017684, -0.0002280, 0.0001912
4: -0.0021069, -0.0015304, -0.0021226, -0.0015553, -0.0002070, 0.0002468
5: 0.0055617, 0.0061072, 0.0055851, 0.0061220, -0.0002335, 0.0001959
6: -0.0002333, 0.0019311, -0.0001401, 0.0019901, -0.0009266, 0.0007771
7: -0.0051868, -0.0022390, -0.0052671, -0.0023659, -0.0010584, 0.0012620
8: 0.9855602, 0.9876367, 0.9855036, 0.9875473, -0.0007455, 0.0008890
9: -0.0046647, -0.0027798, -0.0045835, -0.0027284, -0.0008069, 0.0006768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005647
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005554
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0034173, 0.0043486, -0.0004185, 0.0003699
1: 0.0018078, 0.0019517, 0.0018160, 0.0019505, -0.0000605, 0.0000534
2: 0.0119511, 0.0125021, 0.0119557, 0.0124705, -0.0002045, 0.0002314
3: -0.0023200, -0.0017502, -0.0023154, -0.0017829, -0.0002115, 0.0002393
4: -0.0021422, -0.0015254, -0.0021069, -0.0015304, -0.0002591, 0.0002290
5: 0.0055569, 0.0061406, 0.0055617, 0.0061072, -0.0002167, 0.0002452
6: -0.0002522, 0.0020639, -0.0002333, 0.0019311, -0.0008597, 0.0009728
7: -0.0053675, -0.0022132, -0.0051868, -0.0022390, -0.0013248, 0.0011708
8: 0.9854329, 0.9876548, 0.9855602, 0.9876367, -0.0009332, 0.0008247
9: -0.0046812, -0.0026642, -0.0046647, -0.0027798, -0.0007486, 0.0008471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005642
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005553
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033920, 0.0043085, -0.0004422, 0.0004578
1: 0.0018078, 0.0019517, 0.0018123, 0.0019448, -0.0000639, 0.0000661
2: 0.0119511, 0.0125021, 0.0119778, 0.0124845, -0.0002531, 0.0002445
3: -0.0023200, -0.0017502, -0.0022924, -0.0017684, -0.0002618, 0.0002528
4: -0.0021422, -0.0015254, -0.0021226, -0.0015553, -0.0002737, 0.0002834
5: 0.0055569, 0.0061406, 0.0055851, 0.0061220, -0.0002682, 0.0002590
6: -0.0002522, 0.0020639, -0.0001401, 0.0019901, -0.0010641, 0.0010277
7: -0.0053675, -0.0022132, -0.0052671, -0.0023659, -0.0013997, 0.0014492
8: 0.9854329, 0.9876548, 0.9855036, 0.9875473, -0.0009859, 0.0010209
9: -0.0046812, -0.0026642, -0.0045835, -0.0027284, -0.0009267, 0.0008950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005642
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005553
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033794, 0.0042970, -0.0004008, 0.0004327
1: 0.0018123, 0.0019448, 0.0018105, 0.0019431, -0.0000579, 0.0000625
2: 0.0119778, 0.0124845, 0.0119841, 0.0124915, -0.0002392, 0.0002216
3: -0.0022924, -0.0017684, -0.0022859, -0.0017612, -0.0002474, 0.0002292
4: -0.0021226, -0.0015553, -0.0021303, -0.0015624, -0.0002481, 0.0002678
5: 0.0055851, 0.0061220, 0.0055919, 0.0061294, -0.0002535, 0.0002348
6: -0.0001401, 0.0019901, -0.0001135, 0.0020192, -0.0010057, 0.0009316
7: -0.0052671, -0.0023659, -0.0053067, -0.0024022, -0.0012687, 0.0013697
8: 0.9855036, 0.9875473, 0.9854757, 0.9875218, -0.0008937, 0.0009648
9: -0.0045835, -0.0027284, -0.0045603, -0.0027031, -0.0008758, 0.0008113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005647
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005568
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033497, 0.0042613, -0.0003612, 0.0004591
1: 0.0018123, 0.0019448, 0.0018062, 0.0019379, -0.0000522, 0.0000663
2: 0.0119778, 0.0124845, 0.0120039, 0.0125079, -0.0002538, 0.0001997
3: -0.0022924, -0.0017684, -0.0022654, -0.0017442, -0.0002625, 0.0002065
4: -0.0021226, -0.0015553, -0.0021488, -0.0015845, -0.0002236, 0.0002842
5: 0.0055851, 0.0061220, 0.0056128, 0.0061468, -0.0002690, 0.0002116
6: -0.0001401, 0.0019901, -0.0000304, 0.0020885, -0.0010672, 0.0008395
7: -0.0052671, -0.0023659, -0.0054010, -0.0025153, -0.0011433, 0.0014534
8: 0.9855036, 0.9875473, 0.9854093, 0.9874420, -0.0008054, 0.0010238
9: -0.0045835, -0.0027284, -0.0044880, -0.0026428, -0.0009293, 0.0007311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005647
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005568
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033794, 0.0042970, -0.0004722, 0.0004538
1: 0.0018038, 0.0019463, 0.0018105, 0.0019431, -0.0000682, 0.0000656
2: 0.0119718, 0.0125172, 0.0119841, 0.0124915, -0.0002509, 0.0002610
3: -0.0022986, -0.0017346, -0.0022859, -0.0017612, -0.0002595, 0.0002700
4: -0.0021592, -0.0015485, -0.0021303, -0.0015624, -0.0002923, 0.0002809
5: 0.0055788, 0.0061567, 0.0055919, 0.0061294, -0.0002658, 0.0002766
6: -0.0001653, 0.0021275, -0.0001135, 0.0020192, -0.0010547, 0.0010974
7: -0.0054542, -0.0023316, -0.0053067, -0.0024022, -0.0014946, 0.0014364
8: 0.9853718, 0.9875715, 0.9854757, 0.9875218, -0.0010528, 0.0010118
9: -0.0046055, -0.0026088, -0.0045603, -0.0027031, -0.0009185, 0.0009557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005647
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005568
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033497, 0.0042613, -0.0004364, 0.0004850
1: 0.0018038, 0.0019463, 0.0018062, 0.0019379, -0.0000630, 0.0000701
2: 0.0119718, 0.0125172, 0.0120039, 0.0125079, -0.0002681, 0.0002413
3: -0.0022986, -0.0017346, -0.0022654, -0.0017442, -0.0002773, 0.0002495
4: -0.0021592, -0.0015485, -0.0021488, -0.0015845, -0.0002701, 0.0003002
5: 0.0055788, 0.0061567, 0.0056128, 0.0061468, -0.0002841, 0.0002556
6: -0.0001653, 0.0021275, -0.0000304, 0.0020885, -0.0011272, 0.0010143
7: -0.0054542, -0.0023316, -0.0054010, -0.0025153, -0.0013814, 0.0015352
8: 0.9853718, 0.9875715, 0.9854093, 0.9874420, -0.0009731, 0.0010814
9: -0.0046055, -0.0026088, -0.0044880, -0.0026428, -0.0009816, 0.0008833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005647
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005568
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0034173, 0.0043486, -0.0003987, 0.0003344
1: 0.0018123, 0.0019448, 0.0018160, 0.0019505, -0.0000576, 0.0000483
2: 0.0119778, 0.0124845, 0.0119557, 0.0124705, -0.0001849, 0.0002204
3: -0.0022924, -0.0017684, -0.0023154, -0.0017829, -0.0001912, 0.0002280
4: -0.0021226, -0.0015553, -0.0021069, -0.0015304, -0.0002468, 0.0002070
5: 0.0055851, 0.0061220, 0.0055617, 0.0061072, -0.0001959, 0.0002335
6: -0.0001401, 0.0019901, -0.0002333, 0.0019311, -0.0007771, 0.0009266
7: -0.0052671, -0.0023659, -0.0051868, -0.0022390, -0.0012620, 0.0010584
8: 0.9855036, 0.9875473, 0.9855602, 0.9876367, -0.0008890, 0.0007455
9: -0.0045835, -0.0027284, -0.0046647, -0.0027798, -0.0006768, 0.0008069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005555
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005463
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033920, 0.0043085, -0.0003110, 0.0003110
1: 0.0018123, 0.0019448, 0.0018123, 0.0019448, -0.0000449, 0.0000449
2: 0.0119778, 0.0124845, 0.0119778, 0.0124845, -0.0001719, 0.0001719
3: -0.0022924, -0.0017684, -0.0022924, -0.0017684, -0.0001778, 0.0001778
4: -0.0021226, -0.0015553, -0.0021226, -0.0015553, -0.0001925, 0.0001925
5: 0.0055851, 0.0061220, 0.0055851, 0.0061220, -0.0001822, 0.0001822
6: -0.0001401, 0.0019901, -0.0001401, 0.0019901, -0.0007228, 0.0007228
7: -0.0052671, -0.0023659, -0.0052671, -0.0023659, -0.0009844, 0.0009844
8: 0.9855036, 0.9875473, 0.9855036, 0.9875473, -0.0006935, 0.0006935
9: -0.0045835, -0.0027284, -0.0045835, -0.0027284, -0.0006295, 0.0006295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005555
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005463
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0034173, 0.0043486, -0.0004732, 0.0003586
1: 0.0018038, 0.0019463, 0.0018160, 0.0019505, -0.0000684, 0.0000518
2: 0.0119718, 0.0125172, 0.0119557, 0.0124705, -0.0001983, 0.0002616
3: -0.0022986, -0.0017346, -0.0023154, -0.0017829, -0.0002051, 0.0002706
4: -0.0021592, -0.0015485, -0.0021069, -0.0015304, -0.0002929, 0.0002220
5: 0.0055788, 0.0061567, 0.0055617, 0.0061072, -0.0002101, 0.0002772
6: -0.0001653, 0.0021275, -0.0002333, 0.0019311, -0.0008335, 0.0010999
7: -0.0054542, -0.0023316, -0.0051868, -0.0022390, -0.0014979, 0.0011351
8: 0.9853718, 0.9875715, 0.9855602, 0.9876367, -0.0010552, 0.0007996
9: -0.0046055, -0.0026088, -0.0046647, -0.0027798, -0.0007258, 0.0009578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005554
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005463
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033920, 0.0043085, -0.0004177, 0.0003697
1: 0.0018038, 0.0019463, 0.0018123, 0.0019448, -0.0000603, 0.0000534
2: 0.0119718, 0.0125172, 0.0119778, 0.0124845, -0.0002044, 0.0002309
3: -0.0022986, -0.0017346, -0.0022924, -0.0017684, -0.0002114, 0.0002388
4: -0.0021592, -0.0015485, -0.0021226, -0.0015553, -0.0002586, 0.0002289
5: 0.0055788, 0.0061567, 0.0055851, 0.0061220, -0.0002166, 0.0002447
6: -0.0001653, 0.0021275, -0.0001401, 0.0019901, -0.0008594, 0.0009708
7: -0.0054542, -0.0023316, -0.0052671, -0.0023659, -0.0013222, 0.0011704
8: 0.9853718, 0.9875715, 0.9855036, 0.9875473, -0.0009314, 0.0008245
9: -0.0046055, -0.0026088, -0.0045835, -0.0027284, -0.0007484, 0.0008454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005554
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005463
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033228, 0.0043039, -0.0004219, 0.0005675
1: 0.0018160, 0.0019505, 0.0018024, 0.0019441, -0.0000609, 0.0000820
2: 0.0119557, 0.0124705, 0.0119803, 0.0125228, -0.0003138, 0.0002332
3: -0.0023154, -0.0017829, -0.0022898, -0.0017288, -0.0003245, 0.0002412
4: -0.0021069, -0.0015304, -0.0021654, -0.0015581, -0.0002611, 0.0003513
5: 0.0055617, 0.0061072, 0.0055878, 0.0061625, -0.0003324, 0.0002471
6: -0.0002333, 0.0019311, -0.0001295, 0.0021509, -0.0013190, 0.0009806
7: -0.0051868, -0.0022390, -0.0054860, -0.0023803, -0.0013354, 0.0017964
8: 0.9855602, 0.9876367, 0.9853494, 0.9875371, -0.0009407, 0.0012654
9: -0.0046647, -0.0027798, -0.0045743, -0.0025884, -0.0011487, 0.0008539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005927
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005821
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0032945, 0.0042682, -0.0004119, 0.0006225
1: 0.0018160, 0.0019505, 0.0017983, 0.0019389, -0.0000595, 0.0000899
2: 0.0119557, 0.0124705, 0.0120001, 0.0125384, -0.0003442, 0.0002277
3: -0.0023154, -0.0017829, -0.0022694, -0.0017127, -0.0003560, 0.0002355
4: -0.0021069, -0.0015304, -0.0021829, -0.0015802, -0.0002549, 0.0003854
5: 0.0055617, 0.0061072, 0.0056088, 0.0061791, -0.0003647, 0.0002413
6: -0.0002333, 0.0019311, -0.0000464, 0.0022166, -0.0014469, 0.0009573
7: -0.0051868, -0.0022390, -0.0055755, -0.0024935, -0.0013037, 0.0019706
8: 0.9855602, 0.9876367, 0.9852863, 0.9874574, -0.0009184, 0.0013881
9: -0.0046647, -0.0027798, -0.0045019, -0.0025312, -0.0012601, 0.0008336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005927
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005821
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033228, 0.0043039, -0.0003790, 0.0004768
1: 0.0018078, 0.0019517, 0.0018024, 0.0019441, -0.0000547, 0.0000689
2: 0.0119511, 0.0125021, 0.0119803, 0.0125228, -0.0002636, 0.0002095
3: -0.0023200, -0.0017502, -0.0022898, -0.0017288, -0.0002726, 0.0002167
4: -0.0021422, -0.0015254, -0.0021654, -0.0015581, -0.0002346, 0.0002952
5: 0.0055569, 0.0061406, 0.0055878, 0.0061625, -0.0002793, 0.0002220
6: -0.0002522, 0.0020639, -0.0001295, 0.0021509, -0.0011082, 0.0008808
7: -0.0053675, -0.0022132, -0.0054860, -0.0023803, -0.0011996, 0.0015093
8: 0.9854329, 0.9876548, 0.9853494, 0.9875371, -0.0008450, 0.0010632
9: -0.0046812, -0.0026642, -0.0045743, -0.0025884, -0.0009651, 0.0007671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005778
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005708
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0032945, 0.0042682, -0.0004036, 0.0005614
1: 0.0018078, 0.0019517, 0.0017983, 0.0019389, -0.0000583, 0.0000811
2: 0.0119511, 0.0125021, 0.0120001, 0.0125384, -0.0003104, 0.0002232
3: -0.0023200, -0.0017502, -0.0022694, -0.0017127, -0.0003210, 0.0002308
4: -0.0021422, -0.0015254, -0.0021829, -0.0015802, -0.0002498, 0.0003475
5: 0.0055569, 0.0061406, 0.0056088, 0.0061791, -0.0003289, 0.0002364
6: -0.0002522, 0.0020639, -0.0000464, 0.0022166, -0.0013049, 0.0009381
7: -0.0053675, -0.0022132, -0.0055755, -0.0024935, -0.0012776, 0.0017772
8: 0.9854329, 0.9876548, 0.9852863, 0.9874574, -0.0009000, 0.0012519
9: -0.0046812, -0.0026642, -0.0045019, -0.0025312, -0.0011364, 0.0008170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005778
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005707
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033602, 0.0043567, -0.0003699, 0.0004185
1: 0.0018160, 0.0019505, 0.0018078, 0.0019517, -0.0000534, 0.0000605
2: 0.0119557, 0.0124705, 0.0119511, 0.0125021, -0.0002314, 0.0002045
3: -0.0023154, -0.0017829, -0.0023200, -0.0017502, -0.0002393, 0.0002115
4: -0.0021069, -0.0015304, -0.0021422, -0.0015254, -0.0002290, 0.0002591
5: 0.0055617, 0.0061072, 0.0055569, 0.0061406, -0.0002452, 0.0002167
6: -0.0002333, 0.0019311, -0.0002522, 0.0020639, -0.0009728, 0.0008597
7: -0.0051868, -0.0022390, -0.0053675, -0.0022132, -0.0011708, 0.0013248
8: 0.9855602, 0.9876367, 0.9854329, 0.9876548, -0.0008247, 0.0009332
9: -0.0046647, -0.0027798, -0.0046812, -0.0026642, -0.0008471, 0.0007486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005896
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005794
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034173, 0.0043486, 0.0033329, 0.0043193, -0.0003586, 0.0004732
1: 0.0018160, 0.0019505, 0.0018038, 0.0019463, -0.0000518, 0.0000684
2: 0.0119557, 0.0124705, 0.0119718, 0.0125172, -0.0002616, 0.0001983
3: -0.0023154, -0.0017829, -0.0022986, -0.0017346, -0.0002706, 0.0002051
4: -0.0021069, -0.0015304, -0.0021592, -0.0015485, -0.0002220, 0.0002929
5: 0.0055617, 0.0061072, 0.0055788, 0.0061567, -0.0002772, 0.0002101
6: -0.0002333, 0.0019311, -0.0001653, 0.0021275, -0.0010999, 0.0008335
7: -0.0051868, -0.0022390, -0.0054542, -0.0023316, -0.0011351, 0.0014979
8: 0.9855602, 0.9876367, 0.9853718, 0.9875715, -0.0007996, 0.0010552
9: -0.0046647, -0.0027798, -0.0046055, -0.0026088, -0.0009578, 0.0007258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005896
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005794
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033602, 0.0043567, -0.0003300, 0.0003300
1: 0.0018078, 0.0019517, 0.0018078, 0.0019517, -0.0000477, 0.0000477
2: 0.0119511, 0.0125021, 0.0119511, 0.0125021, -0.0001824, 0.0001824
3: -0.0023200, -0.0017502, -0.0023200, -0.0017502, -0.0001887, 0.0001887
4: -0.0021422, -0.0015254, -0.0021422, -0.0015254, -0.0002042, 0.0002042
5: 0.0055569, 0.0061406, 0.0055569, 0.0061406, -0.0001933, 0.0001933
6: -0.0002522, 0.0020639, -0.0002522, 0.0020639, -0.0007669, 0.0007669
7: -0.0053675, -0.0022132, -0.0053675, -0.0022132, -0.0010445, 0.0010445
8: 0.9854329, 0.9876548, 0.9854329, 0.9876548, -0.0007357, 0.0007357
9: -0.0046812, -0.0026642, -0.0046812, -0.0026642, -0.0006679, 0.0006679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005730
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005660
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033602, 0.0043567, 0.0033329, 0.0043193, -0.0003525, 0.0004150
1: 0.0018078, 0.0019517, 0.0018038, 0.0019463, -0.0000509, 0.0000600
2: 0.0119511, 0.0125021, 0.0119718, 0.0125172, -0.0002295, 0.0001949
3: -0.0023200, -0.0017502, -0.0022986, -0.0017346, -0.0002373, 0.0002016
4: -0.0021422, -0.0015254, -0.0021592, -0.0015485, -0.0002182, 0.0002569
5: 0.0055569, 0.0061406, 0.0055788, 0.0061567, -0.0002431, 0.0002065
6: -0.0002522, 0.0020639, -0.0001653, 0.0021275, -0.0009647, 0.0008193
7: -0.0053675, -0.0022132, -0.0054542, -0.0023316, -0.0011158, 0.0013138
8: 0.9854329, 0.9876548, 0.9853718, 0.9875715, -0.0007860, 0.0009255
9: -0.0046812, -0.0026642, -0.0046055, -0.0026088, -0.0008401, 0.0007134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005730
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005660
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033228, 0.0043039, -0.0004601, 0.0005412
1: 0.0018123, 0.0019448, 0.0018024, 0.0019441, -0.0000665, 0.0000782
2: 0.0119778, 0.0124845, 0.0119803, 0.0125228, -0.0002992, 0.0002544
3: -0.0022924, -0.0017684, -0.0022898, -0.0017288, -0.0003095, 0.0002631
4: -0.0021226, -0.0015553, -0.0021654, -0.0015581, -0.0002848, 0.0003350
5: 0.0055851, 0.0061220, 0.0055878, 0.0061625, -0.0003171, 0.0002695
6: -0.0001401, 0.0019901, -0.0001295, 0.0021509, -0.0012580, 0.0010694
7: -0.0052671, -0.0023659, -0.0054860, -0.0023803, -0.0014564, 0.0017133
8: 0.9855036, 0.9875473, 0.9853494, 0.9875371, -0.0010259, 0.0012069
9: -0.0045835, -0.0027284, -0.0045743, -0.0025884, -0.0010955, 0.0009312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005858
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005767
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0032945, 0.0042682, -0.0004194, 0.0005665
1: 0.0018123, 0.0019448, 0.0017983, 0.0019389, -0.0000606, 0.0000818
2: 0.0119778, 0.0124845, 0.0120001, 0.0125384, -0.0003132, 0.0002319
3: -0.0022924, -0.0017684, -0.0022694, -0.0017127, -0.0003239, 0.0002398
4: -0.0021226, -0.0015553, -0.0021829, -0.0015802, -0.0002596, 0.0003507
5: 0.0055851, 0.0061220, 0.0056088, 0.0061791, -0.0003318, 0.0002457
6: -0.0001401, 0.0019901, -0.0000464, 0.0022166, -0.0013166, 0.0009748
7: -0.0052671, -0.0023659, -0.0055755, -0.0024935, -0.0013276, 0.0017932
8: 0.9855036, 0.9875473, 0.9852863, 0.9874574, -0.0009352, 0.0012631
9: -0.0045835, -0.0027284, -0.0045019, -0.0025312, -0.0011466, 0.0008489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005858
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005767
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033228, 0.0043039, -0.0004159, 0.0004497
1: 0.0018038, 0.0019463, 0.0018024, 0.0019441, -0.0000601, 0.0000650
2: 0.0119718, 0.0125172, 0.0119803, 0.0125228, -0.0002486, 0.0002299
3: -0.0022986, -0.0017346, -0.0022898, -0.0017288, -0.0002571, 0.0002378
4: -0.0021592, -0.0015485, -0.0021654, -0.0015581, -0.0002574, 0.0002784
5: 0.0055788, 0.0061567, 0.0055878, 0.0061625, -0.0002634, 0.0002436
6: -0.0001653, 0.0021275, -0.0001295, 0.0021509, -0.0010452, 0.0009666
7: -0.0054542, -0.0023316, -0.0054860, -0.0023803, -0.0013164, 0.0014235
8: 0.9853718, 0.9875715, 0.9853494, 0.9875371, -0.0009273, 0.0010027
9: -0.0046055, -0.0026088, -0.0045743, -0.0025884, -0.0009102, 0.0008418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005733
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005667
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0032945, 0.0042682, -0.0003764, 0.0004758
1: 0.0018038, 0.0019463, 0.0017983, 0.0019389, -0.0000544, 0.0000687
2: 0.0119718, 0.0125172, 0.0120001, 0.0125384, -0.0002630, 0.0002081
3: -0.0022986, -0.0017346, -0.0022694, -0.0017127, -0.0002720, 0.0002152
4: -0.0021592, -0.0015485, -0.0021829, -0.0015802, -0.0002330, 0.0002945
5: 0.0055788, 0.0061567, 0.0056088, 0.0061791, -0.0002787, 0.0002205
6: -0.0001653, 0.0021275, -0.0000464, 0.0022166, -0.0011058, 0.0008749
7: -0.0054542, -0.0023316, -0.0055755, -0.0024935, -0.0011916, 0.0015060
8: 0.9853718, 0.9875715, 0.9852863, 0.9874574, -0.0008394, 0.0010609
9: -0.0046055, -0.0026088, -0.0045019, -0.0025312, -0.0009630, 0.0007619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005733
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005667
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033602, 0.0043567, -0.0004578, 0.0004422
1: 0.0018123, 0.0019448, 0.0018078, 0.0019517, -0.0000661, 0.0000639
2: 0.0119778, 0.0124845, 0.0119511, 0.0125021, -0.0002445, 0.0002531
3: -0.0022924, -0.0017684, -0.0023200, -0.0017502, -0.0002528, 0.0002618
4: -0.0021226, -0.0015553, -0.0021422, -0.0015254, -0.0002834, 0.0002737
5: 0.0055851, 0.0061220, 0.0055569, 0.0061406, -0.0002590, 0.0002682
6: -0.0001401, 0.0019901, -0.0002522, 0.0020639, -0.0010277, 0.0010641
7: -0.0052671, -0.0023659, -0.0053675, -0.0022132, -0.0014492, 0.0013997
8: 0.9855036, 0.9875473, 0.9854329, 0.9876548, -0.0010209, 0.0009859
9: -0.0045835, -0.0027284, -0.0046812, -0.0026642, -0.0008950, 0.0009267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005822
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005716
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033920, 0.0043085, 0.0033329, 0.0043193, -0.0003697, 0.0004177
1: 0.0018123, 0.0019448, 0.0018038, 0.0019463, -0.0000534, 0.0000603
2: 0.0119778, 0.0124845, 0.0119718, 0.0125172, -0.0002309, 0.0002044
3: -0.0022924, -0.0017684, -0.0022986, -0.0017346, -0.0002388, 0.0002114
4: -0.0021226, -0.0015553, -0.0021592, -0.0015485, -0.0002289, 0.0002586
5: 0.0055851, 0.0061220, 0.0055788, 0.0061567, -0.0002447, 0.0002166
6: -0.0001401, 0.0019901, -0.0001653, 0.0021275, -0.0009708, 0.0008594
7: -0.0052671, -0.0023659, -0.0054542, -0.0023316, -0.0011704, 0.0013222
8: 0.9855036, 0.9875473, 0.9853718, 0.9875715, -0.0008245, 0.0009314
9: -0.0045835, -0.0027284, -0.0046055, -0.0026088, -0.0008454, 0.0007484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005822
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005716
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033602, 0.0043567, -0.0004150, 0.0003525
1: 0.0018038, 0.0019463, 0.0018078, 0.0019517, -0.0000600, 0.0000509
2: 0.0119718, 0.0125172, 0.0119511, 0.0125021, -0.0001949, 0.0002295
3: -0.0022986, -0.0017346, -0.0023200, -0.0017502, -0.0002016, 0.0002373
4: -0.0021592, -0.0015485, -0.0021422, -0.0015254, -0.0002569, 0.0002182
5: 0.0055788, 0.0061567, 0.0055569, 0.0061406, -0.0002065, 0.0002431
6: -0.0001653, 0.0021275, -0.0002522, 0.0020639, -0.0008193, 0.0009647
7: -0.0054542, -0.0023316, -0.0053675, -0.0022132, -0.0013138, 0.0011158
8: 0.9853718, 0.9875715, 0.9854329, 0.9876548, -0.0009255, 0.0007860
9: -0.0046055, -0.0026088, -0.0046812, -0.0026642, -0.0007134, 0.0008401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005650
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005579
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033329, 0.0043193, 0.0033329, 0.0043193, -0.0003301, 0.0003301
1: 0.0018038, 0.0019463, 0.0018038, 0.0019463, -0.0000477, 0.0000477
2: 0.0119718, 0.0125172, 0.0119718, 0.0125172, -0.0001825, 0.0001825
3: -0.0022986, -0.0017346, -0.0022986, -0.0017346, -0.0001888, 0.0001888
4: -0.0021592, -0.0015485, -0.0021592, -0.0015485, -0.0002044, 0.0002044
5: 0.0055788, 0.0061567, 0.0055788, 0.0061567, -0.0001934, 0.0001934
6: -0.0001653, 0.0021275, -0.0001653, 0.0021275, -0.0007673, 0.0007673
7: -0.0054542, -0.0023316, -0.0054542, -0.0023316, -0.0010450, 0.0010450
8: 0.9853718, 0.9875715, 0.9853718, 0.9875715, -0.0007362, 0.0007362
9: -0.0046055, -0.0026088, -0.0046055, -0.0026088, -0.0006682, 0.0006682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005650
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005579
time: 0.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.83 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005751
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005667
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005751
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005667
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005821, upper bound: 0.0005746
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005825, upper bound: 0.0005666
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005821, upper bound: 0.0005746
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005825, upper bound: 0.0005666
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005693
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005594
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005693
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005594
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005763, upper bound: 0.0005686
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005594
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005763, upper bound: 0.0005686
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005594
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005699
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005627
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005699
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005627
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005821, upper bound: 0.0005699
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005825, upper bound: 0.0005627
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005821, upper bound: 0.0005699
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005825, upper bound: 0.0005627
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005597
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005492
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005597
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005492
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005763, upper bound: 0.0005596
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005492
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005763, upper bound: 0.0005596
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005492
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005974
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005877
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005974
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005876
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005881, upper bound: 0.0005785
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005892, upper bound: 0.0005725
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005881, upper bound: 0.0005786
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005892, upper bound: 0.0005725
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005953
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005867
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005953
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005867
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005853, upper bound: 0.0005797
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005857, upper bound: 0.0005723
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005853, upper bound: 0.0005797
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005857, upper bound: 0.0005723
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005906
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005826
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005616, upper bound: 0.0005906
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005618, upper bound: 0.0005826
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005882, upper bound: 0.0005742
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005689
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005882, upper bound: 0.0005742
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005893, upper bound: 0.0005689
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005890
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005757
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005890
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005563, upper bound: 0.0005757
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005853, upper bound: 0.0005717
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005857, upper bound: 0.0005648
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005853, upper bound: 0.0005717
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005857, upper bound: 0.0005648
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005697
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005613
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005697
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005613
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005696
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005613
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005696
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005613
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005647
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005554
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005647
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005554
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005642
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005553
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005642
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005553
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005647
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005568
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005647
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005568
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005647
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005568
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005647
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005568
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005555
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005463
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005555
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005463
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005554
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005463
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005554
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005463
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005927
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005821
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005927
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005821
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005778
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005708
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005778
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005707
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005896
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005794
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005896
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005794
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005730
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005660
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005730
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005660
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005858
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005767
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005858
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005767
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005733
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005667
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005733
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005667
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005822
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005716
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005822
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005716
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005650
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005579
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005650
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005579

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0042892, 0.0033797, 0.0042956, -0.0002999, 0.0002921
1: 0.0018107, 0.0019420, 0.0018106, 0.0019429, -0.0000433, 0.0000422
2: 0.0119885, 0.0124907, 0.0119850, 0.0124913, -0.0001615, 0.0001658
3: -0.0022814, -0.0017620, -0.0022851, -0.0017614, -0.0001671, 0.0001715
4: -0.0021295, -0.0015672, -0.0021302, -0.0015633, -0.0001857, 0.0001808
5: 0.0055965, 0.0061285, 0.0055927, 0.0061292, -0.0001711, 0.0001757
6: -0.0000952, 0.0020160, -0.0001101, 0.0020186, -0.0006790, 0.0006971
7: -0.0053023, -0.0024271, -0.0053059, -0.0024068, -0.0009494, 0.0009248
8: 0.9854788, 0.9875042, 0.9854763, 0.9875185, -0.0006688, 0.0006514
9: -0.0045444, -0.0027059, -0.0045574, -0.0027036, -0.0005913, 0.0006071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005525, upper bound: 0.0005639
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005584, upper bound: 0.0005639
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033629, 0.0042627, 0.0033811, 0.0042816, -0.0003395, 0.0002951
1: 0.0018081, 0.0019381, 0.0018108, 0.0019409, -0.0000490, 0.0000426
2: 0.0120031, 0.0125006, 0.0119927, 0.0124905, -0.0001632, 0.0001877
3: -0.0022663, -0.0017517, -0.0022771, -0.0017622, -0.0001688, 0.0001941
4: -0.0021406, -0.0015836, -0.0021293, -0.0015719, -0.0002102, 0.0001827
5: 0.0056120, 0.0061391, 0.0056009, 0.0061284, -0.0001729, 0.0001989
6: -0.0000337, 0.0020577, -0.0000777, 0.0020153, -0.0006860, 0.0007891
7: -0.0053592, -0.0025108, -0.0053014, -0.0024509, -0.0010747, 0.0009343
8: 0.9854388, 0.9874452, 0.9854794, 0.9874874, -0.0007570, 0.0006581
9: -0.0044909, -0.0026696, -0.0045292, -0.0027065, -0.0005974, 0.0006872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005533, upper bound: 0.0005586
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005586
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0042892, 0.0033499, 0.0042597, -0.0003259, 0.0003801
1: 0.0018107, 0.0019420, 0.0018063, 0.0019377, -0.0000471, 0.0000549
2: 0.0119885, 0.0124907, 0.0120048, 0.0125078, -0.0002101, 0.0001802
3: -0.0022814, -0.0017620, -0.0022646, -0.0017443, -0.0002173, 0.0001863
4: -0.0021295, -0.0015672, -0.0021486, -0.0015854, -0.0002017, 0.0002353
5: 0.0055965, 0.0061285, 0.0056137, 0.0061467, -0.0002226, 0.0001909
6: -0.0000952, 0.0020160, -0.0000268, 0.0020879, -0.0008834, 0.0007574
7: -0.0053023, -0.0024271, -0.0054002, -0.0025202, -0.0010315, 0.0012031
8: 0.9854788, 0.9875042, 0.9854099, 0.9874386, -0.0007266, 0.0008475
9: -0.0045444, -0.0027059, -0.0044849, -0.0026433, -0.0007693, 0.0006596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005445, upper bound: 0.0005620
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005505, upper bound: 0.0005620
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033629, 0.0042627, 0.0033513, 0.0042466, -0.0003604, 0.0003828
1: 0.0018081, 0.0019381, 0.0018065, 0.0019358, -0.0000521, 0.0000553
2: 0.0120031, 0.0125006, 0.0120120, 0.0125070, -0.0002117, 0.0001992
3: -0.0022663, -0.0017517, -0.0022571, -0.0017451, -0.0002189, 0.0002061
4: -0.0021406, -0.0015836, -0.0021478, -0.0015936, -0.0002231, 0.0002370
5: 0.0056120, 0.0061391, 0.0056214, 0.0061459, -0.0002243, 0.0002111
6: -0.0000337, 0.0020577, 0.0000037, 0.0020847, -0.0008898, 0.0008376
7: -0.0053592, -0.0025108, -0.0053959, -0.0025618, -0.0011407, 0.0012118
8: 0.9854388, 0.9874452, 0.9854129, 0.9874093, -0.0008036, 0.0008536
9: -0.0044909, -0.0026696, -0.0044583, -0.0026461, -0.0007749, 0.0007294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0005548
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005505, upper bound: 0.0005548
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033243, 0.0042954, 0.0033797, 0.0042956, -0.0004089, 0.0003499
1: 0.0018026, 0.0019429, 0.0018106, 0.0019429, -0.0000591, 0.0000505
2: 0.0119851, 0.0125220, 0.0119850, 0.0124913, -0.0001934, 0.0002261
3: -0.0022850, -0.0017297, -0.0022851, -0.0017614, -0.0002001, 0.0002338
4: -0.0021645, -0.0015634, -0.0021302, -0.0015633, -0.0002531, 0.0002166
5: 0.0055928, 0.0061617, 0.0055927, 0.0061292, -0.0002050, 0.0002395
6: -0.0001097, 0.0021475, -0.0001101, 0.0020186, -0.0008132, 0.0009504
7: -0.0054814, -0.0024073, -0.0053059, -0.0024068, -0.0012944, 0.0011075
8: 0.9853526, 0.9875180, 0.9854763, 0.9875185, -0.0009118, 0.0007801
9: -0.0045570, -0.0025914, -0.0045574, -0.0027036, -0.0007082, 0.0008277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005759, upper bound: 0.0005638
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005772, upper bound: 0.0005638
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033064, 0.0042703, 0.0033811, 0.0042816, -0.0004395, 0.0003501
1: 0.0018000, 0.0019392, 0.0018108, 0.0019409, -0.0000635, 0.0000506
2: 0.0119989, 0.0125318, 0.0119927, 0.0124905, -0.0001936, 0.0002430
3: -0.0022706, -0.0017194, -0.0022771, -0.0017622, -0.0002002, 0.0002513
4: -0.0021756, -0.0015789, -0.0021293, -0.0015719, -0.0002721, 0.0002167
5: 0.0056075, 0.0061722, 0.0056009, 0.0061284, -0.0002051, 0.0002575
6: -0.0000514, 0.0021890, -0.0000777, 0.0020153, -0.0008138, 0.0010215
7: -0.0055380, -0.0024868, -0.0053014, -0.0024509, -0.0013912, 0.0011084
8: 0.9853128, 0.9874621, 0.9854794, 0.9874874, -0.0009800, 0.0007808
9: -0.0045062, -0.0025552, -0.0045292, -0.0027065, -0.0007087, 0.0008896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005584
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005774, upper bound: 0.0005584
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033243, 0.0042954, 0.0033499, 0.0042597, -0.0004348, 0.0004378
1: 0.0018026, 0.0019429, 0.0018063, 0.0019377, -0.0000628, 0.0000632
2: 0.0119851, 0.0125220, 0.0120048, 0.0125078, -0.0002420, 0.0002404
3: -0.0022850, -0.0017297, -0.0022646, -0.0017443, -0.0002503, 0.0002486
4: -0.0021645, -0.0015634, -0.0021486, -0.0015854, -0.0002692, 0.0002710
5: 0.0055928, 0.0061617, 0.0056137, 0.0061467, -0.0002565, 0.0002547
6: -0.0001097, 0.0021475, -0.0000268, 0.0020879, -0.0010175, 0.0010107
7: -0.0054814, -0.0024073, -0.0054002, -0.0025202, -0.0013765, 0.0013858
8: 0.9853526, 0.9875180, 0.9854099, 0.9874386, -0.0009696, 0.0009762
9: -0.0045570, -0.0025914, -0.0044849, -0.0026433, -0.0008861, 0.0008802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005683, upper bound: 0.0005618
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005690, upper bound: 0.0005618
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033064, 0.0042703, 0.0033513, 0.0042466, -0.0004604, 0.0004378
1: 0.0018000, 0.0019392, 0.0018065, 0.0019358, -0.0000665, 0.0000633
2: 0.0119989, 0.0125318, 0.0120120, 0.0125070, -0.0002421, 0.0002545
3: -0.0022706, -0.0017194, -0.0022571, -0.0017451, -0.0002503, 0.0002632
4: -0.0021756, -0.0015789, -0.0021478, -0.0015936, -0.0002850, 0.0002710
5: 0.0056075, 0.0061722, 0.0056214, 0.0061459, -0.0002565, 0.0002697
6: -0.0000514, 0.0021890, 0.0000037, 0.0020847, -0.0010176, 0.0010700
7: -0.0055380, -0.0024868, -0.0053959, -0.0025618, -0.0014573, 0.0013859
8: 0.9853128, 0.9874621, 0.9854129, 0.9874093, -0.0010266, 0.0009763
9: -0.0045062, -0.0025552, -0.0044583, -0.0026461, -0.0008862, 0.0009318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005686, upper bound: 0.0005548
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005692, upper bound: 0.0005548
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0042892, 0.0034176, 0.0043470, -0.0004531, 0.0003489
1: 0.0018107, 0.0019420, 0.0018160, 0.0019503, -0.0000655, 0.0000504
2: 0.0119885, 0.0124907, 0.0119565, 0.0124703, -0.0001929, 0.0002505
3: -0.0022814, -0.0017620, -0.0023145, -0.0017830, -0.0001995, 0.0002591
4: -0.0021295, -0.0015672, -0.0021067, -0.0015314, -0.0002805, 0.0002160
5: 0.0055965, 0.0061285, 0.0055626, 0.0061070, -0.0002044, 0.0002654
6: -0.0000952, 0.0020160, -0.0002297, 0.0019305, -0.0008108, 0.0010531
7: -0.0053023, -0.0024271, -0.0051858, -0.0022439, -0.0014342, 0.0011043
8: 0.9854788, 0.9875042, 0.9855608, 0.9876332, -0.0010103, 0.0007779
9: -0.0045444, -0.0027059, -0.0046615, -0.0027804, -0.0007061, 0.0009170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005491, upper bound: 0.0005640
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005552, upper bound: 0.0005640
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033629, 0.0042627, 0.0034191, 0.0043338, -0.0004877, 0.0003513
1: 0.0018081, 0.0019381, 0.0018163, 0.0019484, -0.0000705, 0.0000508
2: 0.0120031, 0.0125006, 0.0119638, 0.0124695, -0.0001942, 0.0002696
3: -0.0022663, -0.0017517, -0.0023069, -0.0017839, -0.0002009, 0.0002789
4: -0.0021406, -0.0015836, -0.0021058, -0.0015396, -0.0003019, 0.0002175
5: 0.0056120, 0.0061391, 0.0055703, 0.0061061, -0.0002058, 0.0002857
6: -0.0000337, 0.0020577, -0.0001989, 0.0019270, -0.0008166, 0.0011335
7: -0.0053592, -0.0025108, -0.0051811, -0.0022859, -0.0015438, 0.0011121
8: 0.9854388, 0.9874452, 0.9855642, 0.9876037, -0.0010875, 0.0007834
9: -0.0044909, -0.0026696, -0.0046347, -0.0027834, -0.0007111, 0.0009871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005497, upper bound: 0.0005561
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005553, upper bound: 0.0005561
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0042892, 0.0033923, 0.0043070, -0.0004270, 0.0003870
1: 0.0018107, 0.0019420, 0.0018124, 0.0019445, -0.0000617, 0.0000559
2: 0.0119885, 0.0124907, 0.0119786, 0.0124844, -0.0002140, 0.0002361
3: -0.0022814, -0.0017620, -0.0022916, -0.0017685, -0.0002213, 0.0002442
4: -0.0021295, -0.0015672, -0.0021224, -0.0015562, -0.0002643, 0.0002396
5: 0.0055965, 0.0061285, 0.0055860, 0.0061219, -0.0002267, 0.0002501
6: -0.0000952, 0.0020160, -0.0001367, 0.0019895, -0.0008996, 0.0009924
7: -0.0053023, -0.0024271, -0.0052662, -0.0023706, -0.0013516, 0.0012252
8: 0.9854788, 0.9875042, 0.9855043, 0.9875440, -0.0009521, 0.0008630
9: -0.0045444, -0.0027059, -0.0045805, -0.0027290, -0.0007834, 0.0008642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005358, upper bound: 0.0005535
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005536
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033629, 0.0042627, 0.0033936, 0.0042937, -0.0004612, 0.0003897
1: 0.0018081, 0.0019381, 0.0018126, 0.0019426, -0.0000666, 0.0000563
2: 0.0120031, 0.0125006, 0.0119860, 0.0124836, -0.0002155, 0.0002550
3: -0.0022663, -0.0017517, -0.0022840, -0.0017693, -0.0002229, 0.0002637
4: -0.0021406, -0.0015836, -0.0021216, -0.0015644, -0.0002855, 0.0002412
5: 0.0056120, 0.0061391, 0.0055938, 0.0061211, -0.0002283, 0.0002702
6: -0.0000337, 0.0020577, -0.0001058, 0.0019863, -0.0009058, 0.0010719
7: -0.0053592, -0.0025108, -0.0052618, -0.0024127, -0.0014599, 0.0012337
8: 0.9854388, 0.9874452, 0.9855073, 0.9875144, -0.0010284, 0.0008690
9: -0.0044909, -0.0026696, -0.0045536, -0.0027318, -0.0007888, 0.0009335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005358, upper bound: 0.0005430
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005430
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033243, 0.0042954, 0.0034176, 0.0043470, -0.0005621, 0.0004066
1: 0.0018026, 0.0019429, 0.0018160, 0.0019503, -0.0000812, 0.0000587
2: 0.0119851, 0.0125220, 0.0119565, 0.0124703, -0.0002248, 0.0003107
3: -0.0022850, -0.0017297, -0.0023145, -0.0017830, -0.0002325, 0.0003214
4: -0.0021645, -0.0015634, -0.0021067, -0.0015314, -0.0003479, 0.0002517
5: 0.0055928, 0.0061617, 0.0055626, 0.0061070, -0.0002382, 0.0003293
6: -0.0001097, 0.0021475, -0.0002297, 0.0019305, -0.0009450, 0.0013064
7: -0.0054814, -0.0024073, -0.0051858, -0.0022439, -0.0017792, 0.0012870
8: 0.9853526, 0.9875180, 0.9855608, 0.9876332, -0.0012533, 0.0009066
9: -0.0045570, -0.0025914, -0.0046615, -0.0027804, -0.0008229, 0.0011377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005728, upper bound: 0.0005628
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005740, upper bound: 0.0005628
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033064, 0.0042703, 0.0034191, 0.0043338, -0.0005877, 0.0004063
1: 0.0018000, 0.0019392, 0.0018163, 0.0019484, -0.0000849, 0.0000587
2: 0.0119989, 0.0125318, 0.0119638, 0.0124695, -0.0002246, 0.0003249
3: -0.0022706, -0.0017194, -0.0023069, -0.0017839, -0.0002323, 0.0003361
4: -0.0021756, -0.0015789, -0.0021058, -0.0015396, -0.0003638, 0.0002515
5: 0.0056075, 0.0061722, 0.0055703, 0.0061061, -0.0002380, 0.0003443
6: -0.0000514, 0.0021890, -0.0001989, 0.0019270, -0.0009444, 0.0013660
7: -0.0055380, -0.0024868, -0.0051811, -0.0022859, -0.0018603, 0.0012862
8: 0.9853128, 0.9874621, 0.9855642, 0.9876037, -0.0013105, 0.0009060
9: -0.0045062, -0.0025552, -0.0046347, -0.0027834, -0.0008224, 0.0011896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005736, upper bound: 0.0005558
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005744, upper bound: 0.0005558
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033243, 0.0042954, 0.0033923, 0.0043070, -0.0005360, 0.0004448
1: 0.0018026, 0.0019429, 0.0018124, 0.0019445, -0.0000774, 0.0000643
2: 0.0119851, 0.0125220, 0.0119786, 0.0124844, -0.0002459, 0.0002963
3: -0.0022850, -0.0017297, -0.0022916, -0.0017685, -0.0002543, 0.0003065
4: -0.0021645, -0.0015634, -0.0021224, -0.0015562, -0.0003318, 0.0002753
5: 0.0055928, 0.0061617, 0.0055860, 0.0061219, -0.0002605, 0.0003140
6: -0.0001097, 0.0021475, -0.0001367, 0.0019895, -0.0010337, 0.0012457
7: -0.0054814, -0.0024073, -0.0052662, -0.0023706, -0.0016966, 0.0014079
8: 0.9853526, 0.9875180, 0.9855043, 0.9875440, -0.0011951, 0.0009917
9: -0.0045570, -0.0025914, -0.0045805, -0.0027290, -0.0009002, 0.0010848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005531
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005614, upper bound: 0.0005531
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033064, 0.0042703, 0.0033936, 0.0042937, -0.0005612, 0.0004447
1: 0.0018000, 0.0019392, 0.0018126, 0.0019426, -0.0000811, 0.0000643
2: 0.0119989, 0.0125318, 0.0119860, 0.0124836, -0.0002459, 0.0003103
3: -0.0022706, -0.0017194, -0.0022840, -0.0017693, -0.0002543, 0.0003209
4: -0.0021756, -0.0015789, -0.0021216, -0.0015644, -0.0003474, 0.0002753
5: 0.0056075, 0.0061722, 0.0055938, 0.0061211, -0.0002605, 0.0003287
6: -0.0000514, 0.0021890, -0.0001058, 0.0019863, -0.0010337, 0.0013044
7: -0.0055380, -0.0024868, -0.0052618, -0.0024127, -0.0017764, 0.0014078
8: 0.9853128, 0.9874621, 0.9855073, 0.9875144, -0.0012514, 0.0009917
9: -0.0045062, -0.0025552, -0.0045536, -0.0027318, -0.0009002, 0.0011359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005609, upper bound: 0.0005430
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005615, upper bound: 0.0005430
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033511, 0.0042531, 0.0033797, 0.0042956, -0.0003882, 0.0003176
1: 0.0018064, 0.0019367, 0.0018106, 0.0019429, -0.0000561, 0.0000459
2: 0.0120085, 0.0125071, 0.0119850, 0.0124913, -0.0001756, 0.0002146
3: -0.0022608, -0.0017450, -0.0022851, -0.0017614, -0.0001816, 0.0002220
4: -0.0021479, -0.0015896, -0.0021302, -0.0015633, -0.0002403, 0.0001966
5: 0.0056176, 0.0061460, 0.0055927, 0.0061292, -0.0001860, 0.0002274
6: -0.0000113, 0.0020852, -0.0001101, 0.0020186, -0.0007381, 0.0009023
7: -0.0053966, -0.0025413, -0.0053059, -0.0024068, -0.0012288, 0.0010053
8: 0.9854125, 0.9874237, 0.9854763, 0.9875185, -0.0008656, 0.0007081
9: -0.0044714, -0.0026456, -0.0045574, -0.0027036, -0.0006428, 0.0007857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005490, upper bound: 0.0005567
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005548, upper bound: 0.0005567
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033313, 0.0042283, 0.0033811, 0.0042816, -0.0004134, 0.0003182
1: 0.0018036, 0.0019332, 0.0018108, 0.0019409, -0.0000597, 0.0000460
2: 0.0120222, 0.0125181, 0.0119927, 0.0124905, -0.0001759, 0.0002285
3: -0.0022466, -0.0017337, -0.0022771, -0.0017622, -0.0001820, 0.0002364
4: -0.0021602, -0.0016049, -0.0021293, -0.0015719, -0.0002559, 0.0001970
5: 0.0056321, 0.0061576, 0.0056009, 0.0061284, -0.0001864, 0.0002422
6: 0.0000463, 0.0021312, -0.0000777, 0.0020153, -0.0007397, 0.0009608
7: -0.0054593, -0.0026198, -0.0053014, -0.0024509, -0.0013085, 0.0010074
8: 0.9853682, 0.9873684, 0.9854794, 0.9874874, -0.0009218, 0.0007096
9: -0.0044212, -0.0026055, -0.0045292, -0.0027065, -0.0006441, 0.0008367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005495, upper bound: 0.0005505
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005548, upper bound: 0.0005505
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033511, 0.0042531, 0.0033499, 0.0042597, -0.0003029, 0.0002945
1: 0.0018064, 0.0019367, 0.0018063, 0.0019377, -0.0000438, 0.0000426
2: 0.0120085, 0.0125071, 0.0120048, 0.0125078, -0.0001628, 0.0001675
3: -0.0022608, -0.0017450, -0.0022646, -0.0017443, -0.0001684, 0.0001732
4: -0.0021479, -0.0015896, -0.0021486, -0.0015854, -0.0001875, 0.0001823
5: 0.0056176, 0.0061460, 0.0056137, 0.0061467, -0.0001725, 0.0001775
6: -0.0000113, 0.0020852, -0.0000268, 0.0020879, -0.0006846, 0.0007041
7: -0.0053966, -0.0025413, -0.0054002, -0.0025202, -0.0009589, 0.0009324
8: 0.9854125, 0.9874237, 0.9854099, 0.9874386, -0.0006755, 0.0006568
9: -0.0044714, -0.0026456, -0.0044849, -0.0026433, -0.0005962, 0.0006132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005567
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005498, upper bound: 0.0005567
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033313, 0.0042283, 0.0033513, 0.0042466, -0.0003413, 0.0002988
1: 0.0018036, 0.0019332, 0.0018065, 0.0019358, -0.0000493, 0.0000432
2: 0.0120222, 0.0125181, 0.0120120, 0.0125070, -0.0001652, 0.0001887
3: -0.0022466, -0.0017337, -0.0022571, -0.0017451, -0.0001709, 0.0001951
4: -0.0021602, -0.0016049, -0.0021478, -0.0015936, -0.0002112, 0.0001850
5: 0.0056321, 0.0061576, 0.0056214, 0.0061459, -0.0001750, 0.0001999
6: 0.0000463, 0.0021312, 0.0000037, 0.0020847, -0.0006945, 0.0007932
7: -0.0054593, -0.0026198, -0.0053959, -0.0025618, -0.0010803, 0.0009459
8: 0.9853682, 0.9873684, 0.9854129, 0.9874093, -0.0007610, 0.0006663
9: -0.0044212, -0.0026055, -0.0044583, -0.0026461, -0.0006048, 0.0006907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005442, upper bound: 0.0005505
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005499, upper bound: 0.0005505
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032960, 0.0042595, 0.0033797, 0.0042956, -0.0004639, 0.0003410
1: 0.0017985, 0.0019377, 0.0018106, 0.0019429, -0.0000670, 0.0000493
2: 0.0120049, 0.0125376, 0.0119850, 0.0124913, -0.0001886, 0.0002565
3: -0.0022644, -0.0017135, -0.0022851, -0.0017614, -0.0001950, 0.0002653
4: -0.0021820, -0.0015856, -0.0021302, -0.0015633, -0.0002872, 0.0002111
5: 0.0056138, 0.0061783, 0.0055927, 0.0061292, -0.0001998, 0.0002718
6: -0.0000263, 0.0022133, -0.0001101, 0.0020186, -0.0007927, 0.0010782
7: -0.0055710, -0.0025209, -0.0053059, -0.0024068, -0.0014685, 0.0010795
8: 0.9852896, 0.9874380, 0.9854763, 0.9875185, -0.0010344, 0.0007605
9: -0.0044844, -0.0025341, -0.0045574, -0.0027036, -0.0006903, 0.0009390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005732, upper bound: 0.0005567
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005742, upper bound: 0.0005567
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032763, 0.0042344, 0.0033811, 0.0042816, -0.0004896, 0.0003407
1: 0.0017956, 0.0019340, 0.0018108, 0.0019409, -0.0000707, 0.0000492
2: 0.0120188, 0.0125485, 0.0119927, 0.0124905, -0.0001884, 0.0002707
3: -0.0022501, -0.0017022, -0.0022771, -0.0017622, -0.0001948, 0.0002800
4: -0.0021942, -0.0016011, -0.0021293, -0.0015719, -0.0003031, 0.0002109
5: 0.0056285, 0.0061898, 0.0056009, 0.0061284, -0.0001996, 0.0002868
6: 0.0000321, 0.0022590, -0.0000777, 0.0020153, -0.0007919, 0.0011379
7: -0.0056332, -0.0026004, -0.0053014, -0.0024509, -0.0015498, 0.0010784
8: 0.9852457, 0.9873821, 0.9854794, 0.9874874, -0.0010917, 0.0007597
9: -0.0044336, -0.0024943, -0.0045292, -0.0027065, -0.0006896, 0.0009910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005738, upper bound: 0.0005505
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005744, upper bound: 0.0005505
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032960, 0.0042595, 0.0033499, 0.0042597, -0.0004106, 0.0003524
1: 0.0017985, 0.0019377, 0.0018063, 0.0019377, -0.0000593, 0.0000509
2: 0.0120049, 0.0125376, 0.0120048, 0.0125078, -0.0001948, 0.0002270
3: -0.0022644, -0.0017135, -0.0022646, -0.0017443, -0.0002015, 0.0002348
4: -0.0021820, -0.0015856, -0.0021486, -0.0015854, -0.0002542, 0.0002181
5: 0.0056138, 0.0061783, 0.0056137, 0.0061467, -0.0002064, 0.0002405
6: -0.0000263, 0.0022133, -0.0000268, 0.0020879, -0.0008190, 0.0009544
7: -0.0055710, -0.0025209, -0.0054002, -0.0025202, -0.0012998, 0.0011154
8: 0.9852896, 0.9874380, 0.9854099, 0.9874386, -0.0009156, 0.0007857
9: -0.0044844, -0.0025341, -0.0044849, -0.0026433, -0.0007132, 0.0008311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005679, upper bound: 0.0005567
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005567
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032763, 0.0042344, 0.0033513, 0.0042466, -0.0004403, 0.0003534
1: 0.0017956, 0.0019340, 0.0018065, 0.0019358, -0.0000636, 0.0000511
2: 0.0120188, 0.0125485, 0.0120120, 0.0125070, -0.0001954, 0.0002434
3: -0.0022501, -0.0017022, -0.0022571, -0.0017451, -0.0002021, 0.0002517
4: -0.0021942, -0.0016011, -0.0021478, -0.0015936, -0.0002725, 0.0002187
5: 0.0056285, 0.0061898, 0.0056214, 0.0061459, -0.0002070, 0.0002579
6: 0.0000321, 0.0022590, 0.0000037, 0.0020847, -0.0008213, 0.0010233
7: -0.0056332, -0.0026004, -0.0053959, -0.0025618, -0.0013936, 0.0011186
8: 0.9852457, 0.9873821, 0.9854129, 0.9874093, -0.0009817, 0.0007880
9: -0.0044336, -0.0024943, -0.0044583, -0.0026461, -0.0007153, 0.0008911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005684, upper bound: 0.0005505
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005692, upper bound: 0.0005505
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033511, 0.0042531, 0.0034176, 0.0043470, -0.0005413, 0.0003743
1: 0.0018064, 0.0019367, 0.0018160, 0.0019503, -0.0000782, 0.0000541
2: 0.0120085, 0.0125071, 0.0119565, 0.0124703, -0.0002069, 0.0002993
3: -0.0022608, -0.0017450, -0.0023145, -0.0017830, -0.0002140, 0.0003095
4: -0.0021479, -0.0015896, -0.0021067, -0.0015314, -0.0003351, 0.0002317
5: 0.0056176, 0.0061460, 0.0055626, 0.0061070, -0.0002193, 0.0003171
6: -0.0000113, 0.0020852, -0.0002297, 0.0019305, -0.0008699, 0.0012582
7: -0.0053966, -0.0025413, -0.0051858, -0.0022439, -0.0017136, 0.0011848
8: 0.9854125, 0.9874237, 0.9855608, 0.9876332, -0.0012071, 0.0008346
9: -0.0044714, -0.0026456, -0.0046615, -0.0027804, -0.0007576, 0.0010957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005430
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005477, upper bound: 0.0005430
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033313, 0.0042283, 0.0034191, 0.0043338, -0.0005616, 0.0003744
1: 0.0018036, 0.0019332, 0.0018163, 0.0019484, -0.0000811, 0.0000541
2: 0.0120222, 0.0125181, 0.0119638, 0.0124695, -0.0002070, 0.0003105
3: -0.0022466, -0.0017337, -0.0023069, -0.0017839, -0.0002141, 0.0003211
4: -0.0021602, -0.0016049, -0.0021058, -0.0015396, -0.0003476, 0.0002318
5: 0.0056321, 0.0061576, 0.0055703, 0.0061061, -0.0002193, 0.0003290
6: 0.0000463, 0.0021312, -0.0001989, 0.0019270, -0.0008702, 0.0013052
7: -0.0054593, -0.0026198, -0.0051811, -0.0022859, -0.0017776, 0.0011852
8: 0.9853682, 0.9873684, 0.9855642, 0.9876037, -0.0012522, 0.0008349
9: -0.0044212, -0.0026055, -0.0046347, -0.0027834, -0.0007578, 0.0011367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005313
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005477, upper bound: 0.0005313
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033511, 0.0042531, 0.0033923, 0.0043070, -0.0004533, 0.0003467
1: 0.0018064, 0.0019367, 0.0018124, 0.0019445, -0.0000655, 0.0000501
2: 0.0120085, 0.0125071, 0.0119786, 0.0124844, -0.0001917, 0.0002506
3: -0.0022608, -0.0017450, -0.0022916, -0.0017685, -0.0001982, 0.0002592
4: -0.0021479, -0.0015896, -0.0021224, -0.0015562, -0.0002806, 0.0002146
5: 0.0056176, 0.0061460, 0.0055860, 0.0061219, -0.0002031, 0.0002655
6: -0.0000113, 0.0020852, -0.0001367, 0.0019895, -0.0008058, 0.0010535
7: -0.0053966, -0.0025413, -0.0052662, -0.0023706, -0.0014348, 0.0010974
8: 0.9854125, 0.9874237, 0.9855043, 0.9875440, -0.0010107, 0.0007730
9: -0.0044714, -0.0026456, -0.0045805, -0.0027290, -0.0007017, 0.0009174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005355, upper bound: 0.0005430
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005410, upper bound: 0.0005430
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033313, 0.0042283, 0.0033936, 0.0042937, -0.0004861, 0.0003504
1: 0.0018036, 0.0019332, 0.0018126, 0.0019426, -0.0000702, 0.0000506
2: 0.0120222, 0.0125181, 0.0119860, 0.0124836, -0.0001937, 0.0002687
3: -0.0022466, -0.0017337, -0.0022840, -0.0017693, -0.0002003, 0.0002779
4: -0.0021602, -0.0016049, -0.0021216, -0.0015644, -0.0003009, 0.0002169
5: 0.0056321, 0.0061576, 0.0055938, 0.0061211, -0.0002052, 0.0002847
6: 0.0000463, 0.0021312, -0.0001058, 0.0019863, -0.0008144, 0.0011297
7: -0.0054593, -0.0026198, -0.0052618, -0.0024127, -0.0015386, 0.0011091
8: 0.9853682, 0.9873684, 0.9855073, 0.9875144, -0.0010838, 0.0007813
9: -0.0044212, -0.0026055, -0.0045536, -0.0027318, -0.0007092, 0.0009838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005355, upper bound: 0.0005313
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005410, upper bound: 0.0005313
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032960, 0.0042595, 0.0034176, 0.0043470, -0.0006170, 0.0003978
1: 0.0017985, 0.0019377, 0.0018160, 0.0019503, -0.0000891, 0.0000575
2: 0.0120049, 0.0125376, 0.0119565, 0.0124703, -0.0002199, 0.0003411
3: -0.0022644, -0.0017135, -0.0023145, -0.0017830, -0.0002274, 0.0003528
4: -0.0021820, -0.0015856, -0.0021067, -0.0015314, -0.0003820, 0.0002462
5: 0.0056138, 0.0061783, 0.0055626, 0.0061070, -0.0002330, 0.0003615
6: -0.0000263, 0.0022133, -0.0002297, 0.0019305, -0.0009245, 0.0014342
7: -0.0055710, -0.0025209, -0.0051858, -0.0022439, -0.0019532, 0.0012591
8: 0.9852896, 0.9874380, 0.9855608, 0.9876332, -0.0013759, 0.0008869
9: -0.0044844, -0.0025341, -0.0046615, -0.0027804, -0.0008051, 0.0012490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005672, upper bound: 0.0005430
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005678, upper bound: 0.0005430
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032763, 0.0042344, 0.0034191, 0.0043338, -0.0006378, 0.0003969
1: 0.0017956, 0.0019340, 0.0018163, 0.0019484, -0.0000921, 0.0000573
2: 0.0120188, 0.0125485, 0.0119638, 0.0124695, -0.0002194, 0.0003526
3: -0.0022501, -0.0017022, -0.0023069, -0.0017839, -0.0002269, 0.0003647
4: -0.0021942, -0.0016011, -0.0021058, -0.0015396, -0.0003948, 0.0002457
5: 0.0056285, 0.0061898, 0.0055703, 0.0061061, -0.0002325, 0.0003736
6: 0.0000321, 0.0022590, -0.0001989, 0.0019270, -0.0009224, 0.0014824
7: -0.0056332, -0.0026004, -0.0051811, -0.0022859, -0.0020189, 0.0012563
8: 0.9852457, 0.9873821, 0.9855642, 0.9876037, -0.0014221, 0.0008849
9: -0.0044336, -0.0024943, -0.0046347, -0.0027834, -0.0008033, 0.0012909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005676, upper bound: 0.0005313
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005680, upper bound: 0.0005313
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032960, 0.0042595, 0.0033923, 0.0043070, -0.0005609, 0.0004045
1: 0.0017985, 0.0019377, 0.0018124, 0.0019445, -0.0000810, 0.0000584
2: 0.0120049, 0.0125376, 0.0119786, 0.0124844, -0.0002236, 0.0003101
3: -0.0022644, -0.0017135, -0.0022916, -0.0017685, -0.0002313, 0.0003208
4: -0.0021820, -0.0015856, -0.0021224, -0.0015562, -0.0003472, 0.0002504
5: 0.0056138, 0.0061783, 0.0055860, 0.0061219, -0.0002370, 0.0003286
6: -0.0000263, 0.0022133, -0.0001367, 0.0019895, -0.0009402, 0.0013038
7: -0.0055710, -0.0025209, -0.0052662, -0.0023706, -0.0017757, 0.0012804
8: 0.9852896, 0.9874380, 0.9855043, 0.9875440, -0.0012508, 0.0009020
9: -0.0044844, -0.0025341, -0.0045805, -0.0027290, -0.0008187, 0.0011354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005430
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005612, upper bound: 0.0005430
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032763, 0.0042344, 0.0033936, 0.0042937, -0.0005851, 0.0004049
1: 0.0017956, 0.0019340, 0.0018126, 0.0019426, -0.0000845, 0.0000585
2: 0.0120188, 0.0125485, 0.0119860, 0.0124836, -0.0002239, 0.0003235
3: -0.0022501, -0.0017022, -0.0022840, -0.0017693, -0.0002315, 0.0003345
4: -0.0021942, -0.0016011, -0.0021216, -0.0015644, -0.0003622, 0.0002507
5: 0.0056285, 0.0061898, 0.0055938, 0.0061211, -0.0002372, 0.0003427
6: 0.0000321, 0.0022590, -0.0001058, 0.0019863, -0.0009412, 0.0013598
7: -0.0056332, -0.0026004, -0.0052618, -0.0024127, -0.0018520, 0.0012818
8: 0.9852457, 0.9873821, 0.9855073, 0.9875144, -0.0013046, 0.0009029
9: -0.0044336, -0.0024943, -0.0045536, -0.0027318, -0.0008196, 0.0011842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005609, upper bound: 0.0005313
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005614, upper bound: 0.0005313
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0042892, 0.0033231, 0.0043023, -0.0003593, 0.0004008
1: 0.0018107, 0.0019420, 0.0018024, 0.0019439, -0.0000519, 0.0000579
2: 0.0119885, 0.0124907, 0.0119812, 0.0125226, -0.0002216, 0.0001986
3: -0.0022814, -0.0017620, -0.0022889, -0.0017290, -0.0002292, 0.0002054
4: -0.0021295, -0.0015672, -0.0021652, -0.0015591, -0.0002224, 0.0002481
5: 0.0055965, 0.0061285, 0.0055887, 0.0061624, -0.0002348, 0.0002105
6: -0.0000952, 0.0020160, -0.0001258, 0.0021503, -0.0009315, 0.0008350
7: -0.0053023, -0.0024271, -0.0054852, -0.0023853, -0.0011373, 0.0012686
8: 0.9854788, 0.9875042, 0.9853500, 0.9875336, -0.0008011, 0.0008936
9: -0.0045444, -0.0027059, -0.0045711, -0.0025890, -0.0008112, 0.0007272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005518, upper bound: 0.0005847
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005582, upper bound: 0.0005847
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033629, 0.0042627, 0.0033246, 0.0042894, -0.0003949, 0.0004032
1: 0.0018081, 0.0019381, 0.0018026, 0.0019420, -0.0000571, 0.0000582
2: 0.0120031, 0.0125006, 0.0119884, 0.0125218, -0.0002229, 0.0002183
3: -0.0022663, -0.0017517, -0.0022815, -0.0017298, -0.0002305, 0.0002258
4: -0.0021406, -0.0015836, -0.0021643, -0.0015671, -0.0002445, 0.0002496
5: 0.0056120, 0.0061391, 0.0055964, 0.0061615, -0.0002362, 0.0002314
6: -0.0000337, 0.0020577, -0.0000956, 0.0021468, -0.0009371, 0.0009179
7: -0.0053592, -0.0025108, -0.0054805, -0.0024264, -0.0012502, 0.0012763
8: 0.9854388, 0.9874452, 0.9853533, 0.9875046, -0.0008806, 0.0008990
9: -0.0044909, -0.0026696, -0.0045448, -0.0025920, -0.0008161, 0.0007994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005524, upper bound: 0.0005774
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005584, upper bound: 0.0005774
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0042892, 0.0032948, 0.0042666, -0.0003494, 0.0004558
1: 0.0018107, 0.0019420, 0.0017983, 0.0019387, -0.0000505, 0.0000659
2: 0.0119885, 0.0124907, 0.0120010, 0.0125383, -0.0002520, 0.0001932
3: -0.0022814, -0.0017620, -0.0022685, -0.0017128, -0.0002606, 0.0001998
4: -0.0021295, -0.0015672, -0.0021828, -0.0015812, -0.0002163, 0.0002822
5: 0.0055965, 0.0061285, 0.0056097, 0.0061790, -0.0002670, 0.0002047
6: -0.0000952, 0.0020160, -0.0000427, 0.0022160, -0.0010594, 0.0008121
7: -0.0053023, -0.0024271, -0.0055747, -0.0024986, -0.0011061, 0.0014428
8: 0.9854788, 0.9875042, 0.9852869, 0.9874538, -0.0007791, 0.0010164
9: -0.0045444, -0.0027059, -0.0044987, -0.0025317, -0.0009226, 0.0007072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005445, upper bound: 0.0005833
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005505, upper bound: 0.0005833
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033629, 0.0042627, 0.0032962, 0.0042536, -0.0003837, 0.0004582
1: 0.0018081, 0.0019381, 0.0017985, 0.0019368, -0.0000554, 0.0000662
2: 0.0120031, 0.0125006, 0.0120082, 0.0125375, -0.0002533, 0.0002121
3: -0.0022663, -0.0017517, -0.0022611, -0.0017136, -0.0002620, 0.0002194
4: -0.0021406, -0.0015836, -0.0021819, -0.0015892, -0.0002375, 0.0002837
5: 0.0056120, 0.0061391, 0.0056173, 0.0061781, -0.0002684, 0.0002248
6: -0.0000337, 0.0020577, -0.0000126, 0.0022126, -0.0010651, 0.0008919
7: -0.0053592, -0.0025108, -0.0055701, -0.0025396, -0.0012147, 0.0014505
8: 0.9854388, 0.9874452, 0.9852901, 0.9874250, -0.0008556, 0.0010218
9: -0.0044909, -0.0026696, -0.0044725, -0.0025347, -0.0009275, 0.0007767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0005744
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005505, upper bound: 0.0005744
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033243, 0.0042954, 0.0033231, 0.0043023, -0.0003179, 0.0003100
1: 0.0018026, 0.0019429, 0.0018024, 0.0019439, -0.0000459, 0.0000448
2: 0.0119851, 0.0125220, 0.0119812, 0.0125226, -0.0001714, 0.0001757
3: -0.0022850, -0.0017297, -0.0022889, -0.0017290, -0.0001773, 0.0001818
4: -0.0021645, -0.0015634, -0.0021652, -0.0015591, -0.0001968, 0.0001919
5: 0.0055928, 0.0061617, 0.0055887, 0.0061624, -0.0001816, 0.0001862
6: -0.0001097, 0.0021475, -0.0001258, 0.0021503, -0.0007205, 0.0007388
7: -0.0054814, -0.0024073, -0.0054852, -0.0023853, -0.0010062, 0.0009813
8: 0.9853526, 0.9875180, 0.9853500, 0.9875336, -0.0007088, 0.0006912
9: -0.0045570, -0.0025914, -0.0045711, -0.0025890, -0.0006275, 0.0006434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005779, upper bound: 0.0005657
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005797, upper bound: 0.0005658
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033064, 0.0042703, 0.0033246, 0.0042894, -0.0003557, 0.0003125
1: 0.0018000, 0.0019392, 0.0018026, 0.0019420, -0.0000514, 0.0000452
2: 0.0119989, 0.0125318, 0.0119884, 0.0125218, -0.0001728, 0.0001967
3: -0.0022706, -0.0017194, -0.0022815, -0.0017298, -0.0001787, 0.0002034
4: -0.0021756, -0.0015789, -0.0021643, -0.0015671, -0.0002202, 0.0001935
5: 0.0056075, 0.0061722, 0.0055964, 0.0061615, -0.0001831, 0.0002084
6: -0.0000514, 0.0021890, -0.0000956, 0.0021468, -0.0007264, 0.0008268
7: -0.0055380, -0.0024868, -0.0054805, -0.0024264, -0.0011260, 0.0009894
8: 0.9853128, 0.9874621, 0.9853533, 0.9875046, -0.0007932, 0.0006969
9: -0.0045062, -0.0025552, -0.0045448, -0.0025920, -0.0006326, 0.0007200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005786, upper bound: 0.0005608
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005799, upper bound: 0.0005609
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033243, 0.0042954, 0.0032948, 0.0042666, -0.0003428, 0.0003947
1: 0.0018026, 0.0019429, 0.0017983, 0.0019387, -0.0000495, 0.0000570
2: 0.0119851, 0.0125220, 0.0120010, 0.0125383, -0.0002182, 0.0001895
3: -0.0022850, -0.0017297, -0.0022685, -0.0017128, -0.0002257, 0.0001960
4: -0.0021645, -0.0015634, -0.0021828, -0.0015812, -0.0002122, 0.0002443
5: 0.0055928, 0.0061617, 0.0056097, 0.0061790, -0.0002312, 0.0002008
6: -0.0001097, 0.0021475, -0.0000427, 0.0022160, -0.0009174, 0.0007967
7: -0.0054814, -0.0024073, -0.0055747, -0.0024986, -0.0010850, 0.0012494
8: 0.9853526, 0.9875180, 0.9852869, 0.9874538, -0.0007643, 0.0008801
9: -0.0045570, -0.0025914, -0.0044987, -0.0025317, -0.0007989, 0.0006938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005722, upper bound: 0.0005652
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005738, upper bound: 0.0005652
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033064, 0.0042703, 0.0032962, 0.0042536, -0.0003756, 0.0003970
1: 0.0018000, 0.0019392, 0.0017985, 0.0019368, -0.0000543, 0.0000574
2: 0.0119989, 0.0125318, 0.0120082, 0.0125375, -0.0002195, 0.0002077
3: -0.0022706, -0.0017194, -0.0022611, -0.0017136, -0.0002270, 0.0002148
4: -0.0021756, -0.0015789, -0.0021819, -0.0015892, -0.0002325, 0.0002458
5: 0.0056075, 0.0061722, 0.0056173, 0.0061781, -0.0002326, 0.0002200
6: -0.0000514, 0.0021890, -0.0000126, 0.0022126, -0.0009227, 0.0008730
7: -0.0055380, -0.0024868, -0.0055701, -0.0025396, -0.0011890, 0.0012567
8: 0.9853128, 0.9874621, 0.9852901, 0.9874250, -0.0008375, 0.0008852
9: -0.0045062, -0.0025552, -0.0044725, -0.0025347, -0.0008036, 0.0007603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005727, upper bound: 0.0005601
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005747, upper bound: 0.0005601
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0042892, 0.0033605, 0.0043550, -0.0004794, 0.0004231
1: 0.0018107, 0.0019420, 0.0018078, 0.0019515, -0.0000693, 0.0000611
2: 0.0119885, 0.0124907, 0.0119521, 0.0125019, -0.0002339, 0.0002651
3: -0.0022814, -0.0017620, -0.0023191, -0.0017504, -0.0002419, 0.0002742
4: -0.0021295, -0.0015672, -0.0021421, -0.0015264, -0.0002968, 0.0002619
5: 0.0055965, 0.0061285, 0.0055579, 0.0061405, -0.0002478, 0.0002809
6: -0.0000952, 0.0020160, -0.0002483, 0.0020632, -0.0009834, 0.0011144
7: -0.0053023, -0.0024271, -0.0053666, -0.0022186, -0.0015177, 0.0013392
8: 0.9854788, 0.9875042, 0.9854335, 0.9876511, -0.0010691, 0.0009434
9: -0.0045444, -0.0027059, -0.0046777, -0.0026648, -0.0008563, 0.0009704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005874
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005552, upper bound: 0.0005874
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033629, 0.0042627, 0.0033620, 0.0043433, -0.0005136, 0.0004255
1: 0.0018081, 0.0019381, 0.0018080, 0.0019498, -0.0000742, 0.0000615
2: 0.0120031, 0.0125006, 0.0119585, 0.0125011, -0.0002352, 0.0002840
3: -0.0022663, -0.0017517, -0.0023124, -0.0017513, -0.0002433, 0.0002937
4: -0.0021406, -0.0015836, -0.0021411, -0.0015337, -0.0003179, 0.0002634
5: 0.0056120, 0.0061391, 0.0055647, 0.0061396, -0.0002492, 0.0003009
6: -0.0000337, 0.0020577, -0.0002211, 0.0020597, -0.0009889, 0.0011938
7: -0.0053592, -0.0025108, -0.0053618, -0.0022556, -0.0016259, 0.0013468
8: 0.9854388, 0.9874452, 0.9854369, 0.9876250, -0.0011453, 0.0009487
9: -0.0044909, -0.0026696, -0.0046541, -0.0026678, -0.0008612, 0.0010396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005493, upper bound: 0.0005814
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005552, upper bound: 0.0005814
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033809, 0.0042892, 0.0033332, 0.0043177, -0.0004477, 0.0004584
1: 0.0018107, 0.0019420, 0.0018038, 0.0019461, -0.0000647, 0.0000662
2: 0.0119885, 0.0124907, 0.0119727, 0.0125170, -0.0002534, 0.0002475
3: -0.0022814, -0.0017620, -0.0022977, -0.0017347, -0.0002621, 0.0002560
4: -0.0021295, -0.0015672, -0.0021590, -0.0015496, -0.0002771, 0.0002838
5: 0.0055965, 0.0061285, 0.0055798, 0.0061565, -0.0002685, 0.0002622
6: -0.0000952, 0.0020160, -0.0001614, 0.0021268, -0.0010655, 0.0010405
7: -0.0053023, -0.0024271, -0.0054533, -0.0023368, -0.0014171, 0.0014511
8: 0.9854788, 0.9875042, 0.9853725, 0.9875678, -0.0009982, 0.0010222
9: -0.0045444, -0.0027059, -0.0046021, -0.0026094, -0.0009278, 0.0009061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005358, upper bound: 0.0005797
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005797
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033629, 0.0042627, 0.0033346, 0.0043055, -0.0004837, 0.0004607
1: 0.0018081, 0.0019381, 0.0018041, 0.0019443, -0.0000699, 0.0000666
2: 0.0120031, 0.0125006, 0.0119795, 0.0125163, -0.0002547, 0.0002674
3: -0.0022663, -0.0017517, -0.0022907, -0.0017356, -0.0002634, 0.0002766
4: -0.0021406, -0.0015836, -0.0021581, -0.0015571, -0.0002994, 0.0002852
5: 0.0056120, 0.0061391, 0.0055869, 0.0061557, -0.0002699, 0.0002833
6: -0.0000337, 0.0020577, -0.0001331, 0.0021235, -0.0010708, 0.0011242
7: -0.0053592, -0.0025108, -0.0054487, -0.0023754, -0.0015311, 0.0014583
8: 0.9854388, 0.9874452, 0.9853757, 0.9875406, -0.0010785, 0.0010273
9: -0.0044909, -0.0026696, -0.0045774, -0.0026123, -0.0009325, 0.0009790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005358, upper bound: 0.0005693
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005693
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033243, 0.0042954, 0.0033605, 0.0043550, -0.0004710, 0.0003651
1: 0.0018026, 0.0019429, 0.0018078, 0.0019515, -0.0000680, 0.0000528
2: 0.0119851, 0.0125220, 0.0119521, 0.0125019, -0.0002019, 0.0002604
3: -0.0022850, -0.0017297, -0.0023191, -0.0017504, -0.0002088, 0.0002693
4: -0.0021645, -0.0015634, -0.0021421, -0.0015264, -0.0002915, 0.0002260
5: 0.0055928, 0.0061617, 0.0055579, 0.0061405, -0.0002139, 0.0002759
6: -0.0001097, 0.0021475, -0.0002483, 0.0020632, -0.0008487, 0.0010947
7: -0.0054814, -0.0024073, -0.0053666, -0.0022186, -0.0014908, 0.0011558
8: 0.9853526, 0.9875180, 0.9854335, 0.9876511, -0.0010502, 0.0008142
9: -0.0045570, -0.0025914, -0.0046777, -0.0026648, -0.0007391, 0.0009533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005773, upper bound: 0.0005684
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0005684
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033064, 0.0042703, 0.0033620, 0.0043433, -0.0005040, 0.0003671
1: 0.0018000, 0.0019392, 0.0018080, 0.0019498, -0.0000728, 0.0000530
2: 0.0119989, 0.0125318, 0.0119585, 0.0125011, -0.0002030, 0.0002786
3: -0.0022706, -0.0017194, -0.0023124, -0.0017513, -0.0002099, 0.0002882
4: -0.0021756, -0.0015789, -0.0021411, -0.0015337, -0.0003120, 0.0002273
5: 0.0056075, 0.0061722, 0.0055647, 0.0061396, -0.0002151, 0.0002952
6: -0.0000514, 0.0021890, -0.0002211, 0.0020597, -0.0008533, 0.0011714
7: -0.0055380, -0.0024868, -0.0053618, -0.0022556, -0.0015954, 0.0011622
8: 0.9853128, 0.9874621, 0.9854369, 0.9876250, -0.0011238, 0.0008187
9: -0.0045062, -0.0025552, -0.0046541, -0.0026678, -0.0007431, 0.0010201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005779, upper bound: 0.0005631
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005792, upper bound: 0.0005631
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033243, 0.0042954, 0.0033332, 0.0043177, -0.0004440, 0.0004020
1: 0.0018026, 0.0019429, 0.0018038, 0.0019461, -0.0000641, 0.0000581
2: 0.0119851, 0.0125220, 0.0119727, 0.0125170, -0.0002223, 0.0002455
3: -0.0022850, -0.0017297, -0.0022977, -0.0017347, -0.0002299, 0.0002539
4: -0.0021645, -0.0015634, -0.0021590, -0.0015496, -0.0002748, 0.0002489
5: 0.0055928, 0.0061617, 0.0055798, 0.0061565, -0.0002355, 0.0002601
6: -0.0001097, 0.0021475, -0.0001614, 0.0021268, -0.0009344, 0.0010320
7: -0.0054814, -0.0024073, -0.0054533, -0.0023368, -0.0014054, 0.0012725
8: 0.9853526, 0.9875180, 0.9853725, 0.9875678, -0.0009900, 0.0008964
9: -0.0045570, -0.0025914, -0.0046021, -0.0026094, -0.0008137, 0.0008987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005698, upper bound: 0.0005664
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005714, upper bound: 0.0005664
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033064, 0.0042703, 0.0033346, 0.0043055, -0.0004766, 0.0004042
1: 0.0018000, 0.0019392, 0.0018041, 0.0019443, -0.0000689, 0.0000584
2: 0.0119989, 0.0125318, 0.0119795, 0.0125163, -0.0002235, 0.0002635
3: -0.0022706, -0.0017194, -0.0022907, -0.0017356, -0.0002312, 0.0002725
4: -0.0021756, -0.0015789, -0.0021581, -0.0015571, -0.0002950, 0.0002502
5: 0.0056075, 0.0061722, 0.0055869, 0.0061557, -0.0002368, 0.0002792
6: -0.0000514, 0.0021890, -0.0001331, 0.0021235, -0.0009396, 0.0011078
7: -0.0055380, -0.0024868, -0.0054487, -0.0023754, -0.0015087, 0.0012796
8: 0.9853128, 0.9874621, 0.9853757, 0.9875406, -0.0010628, 0.0009014
9: -0.0045062, -0.0025552, -0.0045774, -0.0026123, -0.0008182, 0.0009647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005701, upper bound: 0.0005592
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005716, upper bound: 0.0005592
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033511, 0.0042531, 0.0033231, 0.0043023, -0.0004475, 0.0004262
1: 0.0018064, 0.0019367, 0.0018024, 0.0019439, -0.0000647, 0.0000616
2: 0.0120085, 0.0125071, 0.0119812, 0.0125226, -0.0002356, 0.0002474
3: -0.0022608, -0.0017450, -0.0022889, -0.0017290, -0.0002437, 0.0002559
4: -0.0021479, -0.0015896, -0.0021652, -0.0015591, -0.0002770, 0.0002638
5: 0.0056176, 0.0061460, 0.0055887, 0.0061624, -0.0002497, 0.0002622
6: -0.0000113, 0.0020852, -0.0001258, 0.0021503, -0.0009906, 0.0010402
7: -0.0053966, -0.0025413, -0.0054852, -0.0023853, -0.0014167, 0.0013491
8: 0.9854125, 0.9874237, 0.9853500, 0.9875336, -0.0009979, 0.0009503
9: -0.0044714, -0.0026456, -0.0045711, -0.0025890, -0.0008626, 0.0009058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005488, upper bound: 0.0005773
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005547, upper bound: 0.0005773
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033313, 0.0042283, 0.0033246, 0.0042894, -0.0004688, 0.0004263
1: 0.0018036, 0.0019332, 0.0018026, 0.0019420, -0.0000677, 0.0000616
2: 0.0120222, 0.0125181, 0.0119884, 0.0125218, -0.0002357, 0.0002592
3: -0.0022466, -0.0017337, -0.0022815, -0.0017298, -0.0002438, 0.0002681
4: -0.0021602, -0.0016049, -0.0021643, -0.0015671, -0.0002902, 0.0002639
5: 0.0056321, 0.0061576, 0.0055964, 0.0061615, -0.0002497, 0.0002746
6: 0.0000463, 0.0021312, -0.0000956, 0.0021468, -0.0009908, 0.0010896
7: -0.0054593, -0.0026198, -0.0054805, -0.0024264, -0.0014840, 0.0013494
8: 0.9853682, 0.9873684, 0.9853533, 0.9875046, -0.0010454, 0.0009505
9: -0.0044212, -0.0026055, -0.0045448, -0.0025920, -0.0008628, 0.0009489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005491, upper bound: 0.0005692
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005548, upper bound: 0.0005692
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033511, 0.0042531, 0.0032948, 0.0042666, -0.0003613, 0.0004020
1: 0.0018064, 0.0019367, 0.0017983, 0.0019387, -0.0000522, 0.0000581
2: 0.0120085, 0.0125071, 0.0120010, 0.0125383, -0.0002222, 0.0001998
3: -0.0022608, -0.0017450, -0.0022685, -0.0017128, -0.0002298, 0.0002066
4: -0.0021479, -0.0015896, -0.0021828, -0.0015812, -0.0002237, 0.0002488
5: 0.0056176, 0.0061460, 0.0056097, 0.0061790, -0.0002355, 0.0002117
6: -0.0000113, 0.0020852, -0.0000427, 0.0022160, -0.0009343, 0.0008398
7: -0.0053966, -0.0025413, -0.0055747, -0.0024986, -0.0011438, 0.0012724
8: 0.9854125, 0.9874237, 0.9852869, 0.9874538, -0.0008057, 0.0008963
9: -0.0044714, -0.0026456, -0.0044987, -0.0025317, -0.0008136, 0.0007314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005773
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005498, upper bound: 0.0005773
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033313, 0.0042283, 0.0032962, 0.0042536, -0.0003959, 0.0004056
1: 0.0018036, 0.0019332, 0.0017985, 0.0019368, -0.0000572, 0.0000586
2: 0.0120222, 0.0125181, 0.0120082, 0.0125375, -0.0002242, 0.0002189
3: -0.0022466, -0.0017337, -0.0022611, -0.0017136, -0.0002319, 0.0002264
4: -0.0021602, -0.0016049, -0.0021819, -0.0015892, -0.0002450, 0.0002510
5: 0.0056321, 0.0061576, 0.0056173, 0.0061781, -0.0002376, 0.0002319
6: 0.0000463, 0.0021312, -0.0000126, 0.0022126, -0.0009426, 0.0009201
7: -0.0054593, -0.0026198, -0.0055701, -0.0025396, -0.0012531, 0.0012838
8: 0.9853682, 0.9873684, 0.9852901, 0.9874250, -0.0008827, 0.0009043
9: -0.0044212, -0.0026055, -0.0044725, -0.0025347, -0.0008209, 0.0008013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005442, upper bound: 0.0005692
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005499, upper bound: 0.0005692
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032960, 0.0042595, 0.0033231, 0.0043023, -0.0004029, 0.0003344
1: 0.0017985, 0.0019377, 0.0018024, 0.0019439, -0.0000582, 0.0000483
2: 0.0120049, 0.0125376, 0.0119812, 0.0125226, -0.0001849, 0.0002227
3: -0.0022644, -0.0017135, -0.0022889, -0.0017290, -0.0001912, 0.0002304
4: -0.0021820, -0.0015856, -0.0021652, -0.0015591, -0.0002494, 0.0002070
5: 0.0056138, 0.0061783, 0.0055887, 0.0061624, -0.0001959, 0.0002360
6: -0.0000263, 0.0022133, -0.0001258, 0.0021503, -0.0007772, 0.0009364
7: -0.0055710, -0.0025209, -0.0054852, -0.0023853, -0.0012753, 0.0010585
8: 0.9852896, 0.9874380, 0.9853500, 0.9875336, -0.0008983, 0.0007456
9: -0.0044844, -0.0025341, -0.0045711, -0.0025890, -0.0006768, 0.0008154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005775, upper bound: 0.0005607
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005790, upper bound: 0.0005607
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032763, 0.0042344, 0.0033246, 0.0042894, -0.0004271, 0.0003347
1: 0.0017956, 0.0019340, 0.0018026, 0.0019420, -0.0000617, 0.0000484
2: 0.0120188, 0.0125485, 0.0119884, 0.0125218, -0.0001850, 0.0002361
3: -0.0022501, -0.0017022, -0.0022815, -0.0017298, -0.0001914, 0.0002442
4: -0.0021942, -0.0016011, -0.0021643, -0.0015671, -0.0002644, 0.0002072
5: 0.0056285, 0.0061898, 0.0055964, 0.0061615, -0.0001961, 0.0002502
6: 0.0000321, 0.0022590, -0.0000956, 0.0021468, -0.0007779, 0.0009927
7: -0.0056332, -0.0026004, -0.0054805, -0.0024264, -0.0013520, 0.0010595
8: 0.9852457, 0.9873821, 0.9853533, 0.9875046, -0.0009524, 0.0007463
9: -0.0044336, -0.0024943, -0.0045448, -0.0025920, -0.0006775, 0.0008645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005780, upper bound: 0.0005562
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005792, upper bound: 0.0005562
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032960, 0.0042595, 0.0032948, 0.0042666, -0.0003193, 0.0003109
1: 0.0017985, 0.0019377, 0.0017983, 0.0019387, -0.0000461, 0.0000449
2: 0.0120049, 0.0125376, 0.0120010, 0.0125383, -0.0001719, 0.0001766
3: -0.0022644, -0.0017135, -0.0022685, -0.0017128, -0.0001778, 0.0001826
4: -0.0021820, -0.0015856, -0.0021828, -0.0015812, -0.0001977, 0.0001925
5: 0.0056138, 0.0061783, 0.0056097, 0.0061790, -0.0001821, 0.0001871
6: -0.0000263, 0.0022133, -0.0000427, 0.0022160, -0.0007226, 0.0007422
7: -0.0055710, -0.0025209, -0.0055747, -0.0024986, -0.0010109, 0.0009842
8: 0.9852896, 0.9874380, 0.9852869, 0.9874538, -0.0007121, 0.0006933
9: -0.0044844, -0.0025341, -0.0044987, -0.0025317, -0.0006293, 0.0006464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005722, upper bound: 0.0005607
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005738, upper bound: 0.0005607
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032763, 0.0042344, 0.0032962, 0.0042536, -0.0003560, 0.0003142
1: 0.0017956, 0.0019340, 0.0017985, 0.0019368, -0.0000514, 0.0000454
2: 0.0120188, 0.0125485, 0.0120082, 0.0125375, -0.0001737, 0.0001968
3: -0.0022501, -0.0017022, -0.0022611, -0.0017136, -0.0001797, 0.0002036
4: -0.0021942, -0.0016011, -0.0021819, -0.0015892, -0.0002204, 0.0001945
5: 0.0056285, 0.0061898, 0.0056173, 0.0061781, -0.0001840, 0.0002085
6: 0.0000321, 0.0022590, -0.0000126, 0.0022126, -0.0007302, 0.0008274
7: -0.0056332, -0.0026004, -0.0055701, -0.0025396, -0.0011269, 0.0009945
8: 0.9852457, 0.9873821, 0.9852901, 0.9874250, -0.0007938, 0.0007006
9: -0.0044336, -0.0024943, -0.0044725, -0.0025347, -0.0006359, 0.0007206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005727, upper bound: 0.0005561
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005562
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033511, 0.0042531, 0.0033605, 0.0043550, -0.0005677, 0.0004485
1: 0.0018064, 0.0019367, 0.0018078, 0.0019515, -0.0000820, 0.0000648
2: 0.0120085, 0.0125071, 0.0119521, 0.0125019, -0.0002480, 0.0003139
3: -0.0022608, -0.0017450, -0.0023191, -0.0017504, -0.0002565, 0.0003246
4: -0.0021479, -0.0015896, -0.0021421, -0.0015264, -0.0003514, 0.0002776
5: 0.0056176, 0.0061460, 0.0055579, 0.0061405, -0.0002627, 0.0003326
6: -0.0000113, 0.0020852, -0.0002483, 0.0020632, -0.0010424, 0.0013195
7: -0.0053966, -0.0025413, -0.0053666, -0.0022186, -0.0017971, 0.0014197
8: 0.9854125, 0.9874237, 0.9854335, 0.9876511, -0.0012659, 0.0010001
9: -0.0044714, -0.0026456, -0.0046777, -0.0026648, -0.0009078, 0.0011491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005415, upper bound: 0.0005732
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005477, upper bound: 0.0005732
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033313, 0.0042283, 0.0033620, 0.0043433, -0.0005875, 0.0004486
1: 0.0018036, 0.0019332, 0.0018080, 0.0019498, -0.0000849, 0.0000648
2: 0.0120222, 0.0125181, 0.0119585, 0.0125011, -0.0002480, 0.0003248
3: -0.0022466, -0.0017337, -0.0023124, -0.0017513, -0.0002565, 0.0003359
4: -0.0021602, -0.0016049, -0.0021411, -0.0015337, -0.0003637, 0.0002777
5: 0.0056321, 0.0061576, 0.0055647, 0.0061396, -0.0002628, 0.0003442
6: 0.0000463, 0.0021312, -0.0002211, 0.0020597, -0.0010426, 0.0013655
7: -0.0054593, -0.0026198, -0.0053618, -0.0022556, -0.0018597, 0.0014199
8: 0.9853682, 0.9873684, 0.9854369, 0.9876250, -0.0013100, 0.0010002
9: -0.0044212, -0.0026055, -0.0046541, -0.0026678, -0.0009079, 0.0011892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005415, upper bound: 0.0005578
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005477, upper bound: 0.0005578
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033511, 0.0042531, 0.0033332, 0.0043177, -0.0004791, 0.0004219
1: 0.0018064, 0.0019367, 0.0018038, 0.0019461, -0.0000692, 0.0000609
2: 0.0120085, 0.0125071, 0.0119727, 0.0125170, -0.0002332, 0.0002649
3: -0.0022608, -0.0017450, -0.0022977, -0.0017347, -0.0002412, 0.0002740
4: -0.0021479, -0.0015896, -0.0021590, -0.0015496, -0.0002966, 0.0002611
5: 0.0056176, 0.0061460, 0.0055798, 0.0061565, -0.0002471, 0.0002807
6: -0.0000113, 0.0020852, -0.0001614, 0.0021268, -0.0009805, 0.0011136
7: -0.0053966, -0.0025413, -0.0054533, -0.0023368, -0.0015166, 0.0013354
8: 0.9854125, 0.9874237, 0.9853725, 0.9875678, -0.0010684, 0.0009407
9: -0.0044714, -0.0026456, -0.0046021, -0.0026094, -0.0008539, 0.0009698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005355, upper bound: 0.0005732
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005410, upper bound: 0.0005732
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033313, 0.0042283, 0.0033346, 0.0043055, -0.0005123, 0.0004255
1: 0.0018036, 0.0019332, 0.0018041, 0.0019443, -0.0000740, 0.0000615
2: 0.0120222, 0.0125181, 0.0119795, 0.0125163, -0.0002352, 0.0002832
3: -0.0022466, -0.0017337, -0.0022907, -0.0017356, -0.0002433, 0.0002929
4: -0.0021602, -0.0016049, -0.0021581, -0.0015571, -0.0003171, 0.0002634
5: 0.0056321, 0.0061576, 0.0055869, 0.0061557, -0.0002492, 0.0003001
6: 0.0000463, 0.0021312, -0.0001331, 0.0021235, -0.0009889, 0.0011907
7: -0.0054593, -0.0026198, -0.0054487, -0.0023754, -0.0016216, 0.0013468
8: 0.9853682, 0.9873684, 0.9853757, 0.9875406, -0.0011423, 0.0009487
9: -0.0044212, -0.0026055, -0.0045774, -0.0026123, -0.0008612, 0.0010369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005355, upper bound: 0.0005578
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005410, upper bound: 0.0005578
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032960, 0.0042595, 0.0033605, 0.0043550, -0.0005560, 0.0003895
1: 0.0017985, 0.0019377, 0.0018078, 0.0019515, -0.0000803, 0.0000563
2: 0.0120049, 0.0125376, 0.0119521, 0.0125019, -0.0002154, 0.0003074
3: -0.0022644, -0.0017135, -0.0023191, -0.0017504, -0.0002227, 0.0003179
4: -0.0021820, -0.0015856, -0.0021421, -0.0015264, -0.0003441, 0.0002411
5: 0.0056138, 0.0061783, 0.0055579, 0.0061405, -0.0002282, 0.0003257
6: -0.0000263, 0.0022133, -0.0002483, 0.0020632, -0.0009054, 0.0012922
7: -0.0055710, -0.0025209, -0.0053666, -0.0022186, -0.0017599, 0.0012330
8: 0.9852896, 0.9874380, 0.9854335, 0.9876511, -0.0012397, 0.0008686
9: -0.0044844, -0.0025341, -0.0046777, -0.0026648, -0.0007884, 0.0011253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005580
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005771, upper bound: 0.0005580
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032763, 0.0042344, 0.0033620, 0.0043433, -0.0005754, 0.0003893
1: 0.0017956, 0.0019340, 0.0018080, 0.0019498, -0.0000831, 0.0000562
2: 0.0120188, 0.0125485, 0.0119585, 0.0125011, -0.0002152, 0.0003181
3: -0.0022501, -0.0017022, -0.0023124, -0.0017513, -0.0002226, 0.0003290
4: -0.0021942, -0.0016011, -0.0021411, -0.0015337, -0.0003562, 0.0002410
5: 0.0056285, 0.0061898, 0.0055647, 0.0061396, -0.0002280, 0.0003371
6: 0.0000321, 0.0022590, -0.0002211, 0.0020597, -0.0009048, 0.0013373
7: -0.0056332, -0.0026004, -0.0053618, -0.0022556, -0.0018213, 0.0012323
8: 0.9852457, 0.9873821, 0.9854369, 0.9876250, -0.0012830, 0.0008680
9: -0.0044336, -0.0024943, -0.0046541, -0.0026678, -0.0007880, 0.0011646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005762, upper bound: 0.0005506
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005773, upper bound: 0.0005506
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032960, 0.0042595, 0.0033332, 0.0043177, -0.0004699, 0.0003619
1: 0.0017985, 0.0019377, 0.0018038, 0.0019461, -0.0000679, 0.0000523
2: 0.0120049, 0.0125376, 0.0119727, 0.0125170, -0.0002001, 0.0002598
3: -0.0022644, -0.0017135, -0.0022977, -0.0017347, -0.0002070, 0.0002687
4: -0.0021820, -0.0015856, -0.0021590, -0.0015496, -0.0002909, 0.0002240
5: 0.0056138, 0.0061783, 0.0055798, 0.0061565, -0.0002120, 0.0002753
6: -0.0000263, 0.0022133, -0.0001614, 0.0021268, -0.0008412, 0.0010922
7: -0.0055710, -0.0025209, -0.0054533, -0.0023368, -0.0014875, 0.0011457
8: 0.9852896, 0.9874380, 0.9853725, 0.9875678, -0.0010479, 0.0008070
9: -0.0044844, -0.0025341, -0.0046021, -0.0026094, -0.0007326, 0.0009512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005695, upper bound: 0.0005580
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005713, upper bound: 0.0005580
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032763, 0.0042344, 0.0033346, 0.0043055, -0.0005010, 0.0003648
1: 0.0017956, 0.0019340, 0.0018041, 0.0019443, -0.0000724, 0.0000527
2: 0.0120188, 0.0125485, 0.0119795, 0.0125163, -0.0002017, 0.0002770
3: -0.0022501, -0.0017022, -0.0022907, -0.0017356, -0.0002086, 0.0002865
4: -0.0021942, -0.0016011, -0.0021581, -0.0015571, -0.0003102, 0.0002258
5: 0.0056285, 0.0061898, 0.0055869, 0.0061557, -0.0002137, 0.0002935
6: 0.0000321, 0.0022590, -0.0001331, 0.0021235, -0.0008479, 0.0011646
7: -0.0056332, -0.0026004, -0.0054487, -0.0023754, -0.0015860, 0.0011547
8: 0.9852457, 0.9873821, 0.9853757, 0.9875406, -0.0011172, 0.0008134
9: -0.0044336, -0.0024943, -0.0045774, -0.0026123, -0.0007384, 0.0010142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005700, upper bound: 0.0005506
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005715, upper bound: 0.0005506
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0034189, 0.0043403, 0.0033797, 0.0042956, -0.0003571, 0.0004445
1: 0.0018162, 0.0019493, 0.0018106, 0.0019429, -0.0000516, 0.0000642
2: 0.0119602, 0.0124696, 0.0119850, 0.0124913, -0.0002457, 0.0001975
3: -0.0023106, -0.0017838, -0.0022851, -0.0017614, -0.0002542, 0.0002042
4: -0.0021059, -0.0015356, -0.0021302, -0.0015633, -0.0002211, 0.0002751
5: 0.0055665, 0.0061063, 0.0055927, 0.0061292, -0.0002604, 0.0002092
6: -0.0002140, 0.0019275, -0.0001101, 0.0020186, -0.0010331, 0.0008301
7: -0.0051818, -0.0022653, -0.0053059, -0.0024068, -0.0011305, 0.0014070
8: 0.9855636, 0.9876181, 0.9854763, 0.9875185, -0.0007963, 0.0009911
9: -0.0046479, -0.0027829, -0.0045574, -0.0027036, -0.0008997, 0.0007229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005517, upper bound: 0.0005619
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005558, upper bound: 0.0005619
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033984, 0.0043154, 0.0033811, 0.0042816, -0.0003871, 0.0004443
1: 0.0018133, 0.0019457, 0.0018108, 0.0019409, -0.0000559, 0.0000642
2: 0.0119740, 0.0124810, 0.0119927, 0.0124905, -0.0002456, 0.0002140
3: -0.0022964, -0.0017720, -0.0022771, -0.0017622, -0.0002540, 0.0002213
4: -0.0021186, -0.0015510, -0.0021293, -0.0015719, -0.0002396, 0.0002750
5: 0.0055811, 0.0061183, 0.0056009, 0.0061284, -0.0002602, 0.0002267
6: -0.0001561, 0.0019752, -0.0000777, 0.0020153, -0.0010326, 0.0008996
7: -0.0052468, -0.0023441, -0.0053014, -0.0024509, -0.0012252, 0.0014063
8: 0.9855179, 0.9875627, 0.9854794, 0.9874874, -0.0008631, 0.0009906
9: -0.0045974, -0.0027414, -0.0045292, -0.0027065, -0.0008992, 0.0007834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005519, upper bound: 0.0005553
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005561, upper bound: 0.0005553
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034189, 0.0043403, 0.0033499, 0.0042597, -0.0003831, 0.0005324
1: 0.0018162, 0.0019493, 0.0018063, 0.0019377, -0.0000553, 0.0000769
2: 0.0119602, 0.0124696, 0.0120048, 0.0125078, -0.0002944, 0.0002118
3: -0.0023106, -0.0017838, -0.0022646, -0.0017443, -0.0003044, 0.0002190
4: -0.0021059, -0.0015356, -0.0021486, -0.0015854, -0.0002371, 0.0003296
5: 0.0055665, 0.0061063, 0.0056137, 0.0061467, -0.0003119, 0.0002244
6: -0.0002140, 0.0019275, -0.0000268, 0.0020879, -0.0012375, 0.0008903
7: -0.0051818, -0.0022653, -0.0054002, -0.0025202, -0.0012126, 0.0016853
8: 0.9855636, 0.9876181, 0.9854099, 0.9874386, -0.0008542, 0.0011872
9: -0.0046479, -0.0027829, -0.0044849, -0.0026433, -0.0010776, 0.0007754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005564
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005564
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033984, 0.0043154, 0.0033513, 0.0042466, -0.0004079, 0.0005319
1: 0.0018133, 0.0019457, 0.0018065, 0.0019358, -0.0000589, 0.0000768
2: 0.0119740, 0.0124810, 0.0120120, 0.0125070, -0.0002941, 0.0002255
3: -0.0022964, -0.0017720, -0.0022571, -0.0017451, -0.0003042, 0.0002333
4: -0.0021186, -0.0015510, -0.0021478, -0.0015936, -0.0002525, 0.0003293
5: 0.0055811, 0.0061183, 0.0056214, 0.0061459, -0.0003116, 0.0002390
6: -0.0001561, 0.0019752, 0.0000037, 0.0020847, -0.0012364, 0.0009481
7: -0.0052468, -0.0023441, -0.0053959, -0.0025618, -0.0012913, 0.0016838
8: 0.9855179, 0.9875627, 0.9854129, 0.9874093, -0.0009096, 0.0011861
9: -0.0045974, -0.0027414, -0.0044583, -0.0026461, -0.0010767, 0.0008257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005477
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005477
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033618, 0.0043476, 0.0033797, 0.0042956, -0.0004313, 0.0004712
1: 0.0018080, 0.0019504, 0.0018106, 0.0019429, -0.0000623, 0.0000681
2: 0.0119562, 0.0125012, 0.0119850, 0.0124913, -0.0002605, 0.0002384
3: -0.0023148, -0.0017511, -0.0022851, -0.0017614, -0.0002694, 0.0002466
4: -0.0021413, -0.0015310, -0.0021302, -0.0015633, -0.0002670, 0.0002917
5: 0.0055622, 0.0061397, 0.0055927, 0.0061292, -0.0002760, 0.0002526
6: -0.0002311, 0.0020603, -0.0001101, 0.0020186, -0.0010952, 0.0010024
7: -0.0053626, -0.0022420, -0.0053059, -0.0024068, -0.0013652, 0.0014916
8: 0.9854363, 0.9876345, 0.9854763, 0.9875185, -0.0009617, 0.0010507
9: -0.0046627, -0.0026673, -0.0045574, -0.0027036, -0.0009538, 0.0008729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005802, upper bound: 0.0005618
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005812, upper bound: 0.0005618
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033386, 0.0043263, 0.0033811, 0.0042816, -0.0004638, 0.0004707
1: 0.0018046, 0.0019473, 0.0018108, 0.0019409, -0.0000670, 0.0000680
2: 0.0119679, 0.0125140, 0.0119927, 0.0124905, -0.0002602, 0.0002564
3: -0.0023026, -0.0017379, -0.0022771, -0.0017622, -0.0002691, 0.0002652
4: -0.0021556, -0.0015442, -0.0021293, -0.0015719, -0.0002871, 0.0002913
5: 0.0055747, 0.0061533, 0.0056009, 0.0061284, -0.0002757, 0.0002717
6: -0.0001816, 0.0021141, -0.0000777, 0.0020153, -0.0010940, 0.0010781
7: -0.0054359, -0.0023094, -0.0053014, -0.0024509, -0.0014683, 0.0014899
8: 0.9853847, 0.9875870, 0.9854794, 0.9874874, -0.0010343, 0.0010495
9: -0.0046196, -0.0026205, -0.0045292, -0.0027065, -0.0009527, 0.0009388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005807, upper bound: 0.0005553
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005814, upper bound: 0.0005552
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033618, 0.0043476, 0.0033499, 0.0042597, -0.0004572, 0.0005591
1: 0.0018080, 0.0019504, 0.0018063, 0.0019377, -0.0000661, 0.0000808
2: 0.0119562, 0.0125012, 0.0120048, 0.0125078, -0.0003091, 0.0002528
3: -0.0023148, -0.0017511, -0.0022646, -0.0017443, -0.0003197, 0.0002614
4: -0.0021413, -0.0015310, -0.0021486, -0.0015854, -0.0002830, 0.0003461
5: 0.0055622, 0.0061397, 0.0056137, 0.0061467, -0.0003275, 0.0002678
6: -0.0002311, 0.0020603, -0.0000268, 0.0020879, -0.0012996, 0.0010627
7: -0.0053626, -0.0022420, -0.0054002, -0.0025202, -0.0014473, 0.0017699
8: 0.9854363, 0.9876345, 0.9854099, 0.9874386, -0.0010195, 0.0012468
9: -0.0046627, -0.0026673, -0.0044849, -0.0026433, -0.0011317, 0.0009254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005562
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005562
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033386, 0.0043263, 0.0033513, 0.0042466, -0.0004847, 0.0005583
1: 0.0018046, 0.0019473, 0.0018065, 0.0019358, -0.0000700, 0.0000807
2: 0.0119679, 0.0125140, 0.0120120, 0.0125070, -0.0003087, 0.0002680
3: -0.0023026, -0.0017379, -0.0022571, -0.0017451, -0.0003193, 0.0002772
4: -0.0021556, -0.0015442, -0.0021478, -0.0015936, -0.0003000, 0.0003456
5: 0.0055747, 0.0061533, 0.0056214, 0.0061459, -0.0003271, 0.0002839
6: -0.0001816, 0.0021141, 0.0000037, 0.0020847, -0.0012977, 0.0011266
7: -0.0054359, -0.0023094, -0.0053959, -0.0025618, -0.0015343, 0.0017674
8: 0.9853847, 0.9875870, 0.9854129, 0.9874093, -0.0010808, 0.0012450
9: -0.0046196, -0.0026205, -0.0044583, -0.0026461, -0.0011301, 0.0009811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005477
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005477
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0034189, 0.0043403, 0.0034176, 0.0043470, -0.0003047, 0.0002969
1: 0.0018162, 0.0019493, 0.0018160, 0.0019503, -0.0000440, 0.0000429
2: 0.0119602, 0.0124696, 0.0119565, 0.0124703, -0.0001641, 0.0001684
3: -0.0023106, -0.0017838, -0.0023145, -0.0017830, -0.0001697, 0.0001742
4: -0.0021059, -0.0015356, -0.0021067, -0.0015314, -0.0001886, 0.0001838
5: 0.0055665, 0.0061063, 0.0055626, 0.0061070, -0.0001739, 0.0001785
6: -0.0002140, 0.0019275, -0.0002297, 0.0019305, -0.0006900, 0.0007082
7: -0.0051818, -0.0022653, -0.0051858, -0.0022439, -0.0009645, 0.0009397
8: 0.9855636, 0.9876181, 0.9855608, 0.9876332, -0.0006794, 0.0006620
9: -0.0046479, -0.0027829, -0.0046615, -0.0027804, -0.0006009, 0.0006167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005517, upper bound: 0.0005565
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005558, upper bound: 0.0005565
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033984, 0.0043154, 0.0034191, 0.0043338, -0.0003451, 0.0003004
1: 0.0018133, 0.0019457, 0.0018163, 0.0019484, -0.0000499, 0.0000434
2: 0.0119740, 0.0124810, 0.0119638, 0.0124695, -0.0001661, 0.0001908
3: -0.0022964, -0.0017720, -0.0023069, -0.0017839, -0.0001718, 0.0001973
4: -0.0021186, -0.0015510, -0.0021058, -0.0015396, -0.0002136, 0.0001860
5: 0.0055811, 0.0061183, 0.0055703, 0.0061061, -0.0001760, 0.0002022
6: -0.0001561, 0.0019752, -0.0001989, 0.0019270, -0.0006982, 0.0008021
7: -0.0052468, -0.0023441, -0.0051811, -0.0022859, -0.0010924, 0.0009509
8: 0.9855179, 0.9875627, 0.9855642, 0.9876037, -0.0007695, 0.0006698
9: -0.0045974, -0.0027414, -0.0046347, -0.0027834, -0.0006080, 0.0006985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005519, upper bound: 0.0005499
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005561, upper bound: 0.0005499
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034189, 0.0043403, 0.0033923, 0.0043070, -0.0003285, 0.0003849
1: 0.0018162, 0.0019493, 0.0018124, 0.0019445, -0.0000475, 0.0000556
2: 0.0119602, 0.0124696, 0.0119786, 0.0124844, -0.0002128, 0.0001816
3: -0.0023106, -0.0017838, -0.0022916, -0.0017685, -0.0002201, 0.0001878
4: -0.0021059, -0.0015356, -0.0021224, -0.0015562, -0.0002033, 0.0002383
5: 0.0055665, 0.0061063, 0.0055860, 0.0061219, -0.0002255, 0.0001924
6: -0.0002140, 0.0019275, -0.0001367, 0.0019895, -0.0008946, 0.0007635
7: -0.0051818, -0.0022653, -0.0052662, -0.0023706, -0.0010398, 0.0012184
8: 0.9855636, 0.9876181, 0.9855043, 0.9875440, -0.0007324, 0.0008583
9: -0.0046479, -0.0027829, -0.0045805, -0.0027290, -0.0007791, 0.0006649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005505
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005505
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033984, 0.0043154, 0.0033936, 0.0042937, -0.0003642, 0.0003882
1: 0.0018133, 0.0019457, 0.0018126, 0.0019426, -0.0000526, 0.0000561
2: 0.0119740, 0.0124810, 0.0119860, 0.0124836, -0.0002146, 0.0002014
3: -0.0022964, -0.0017720, -0.0022840, -0.0017693, -0.0002220, 0.0002083
4: -0.0021186, -0.0015510, -0.0021216, -0.0015644, -0.0002255, 0.0002403
5: 0.0055811, 0.0061183, 0.0055938, 0.0061211, -0.0002274, 0.0002134
6: -0.0001561, 0.0019752, -0.0001058, 0.0019863, -0.0009023, 0.0008466
7: -0.0052468, -0.0023441, -0.0052618, -0.0024127, -0.0011530, 0.0012288
8: 0.9855179, 0.9875627, 0.9855073, 0.9875144, -0.0008122, 0.0008656
9: -0.0045974, -0.0027414, -0.0045536, -0.0027318, -0.0007858, 0.0007372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005404
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005404
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033618, 0.0043476, 0.0034176, 0.0043470, -0.0004129, 0.0003545
1: 0.0018080, 0.0019504, 0.0018160, 0.0019503, -0.0000597, 0.0000512
2: 0.0119562, 0.0125012, 0.0119565, 0.0124703, -0.0001960, 0.0002283
3: -0.0023148, -0.0017511, -0.0023145, -0.0017830, -0.0002027, 0.0002361
4: -0.0021413, -0.0015310, -0.0021067, -0.0015314, -0.0002556, 0.0002194
5: 0.0055622, 0.0061397, 0.0055626, 0.0061070, -0.0002076, 0.0002419
6: -0.0002311, 0.0020603, -0.0002297, 0.0019305, -0.0008239, 0.0009598
7: -0.0053626, -0.0022420, -0.0051858, -0.0022439, -0.0013071, 0.0011221
8: 0.9854363, 0.9876345, 0.9855608, 0.9876332, -0.0009208, 0.0007904
9: -0.0046627, -0.0026673, -0.0046615, -0.0027804, -0.0007175, 0.0008358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005802, upper bound: 0.0005563
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005812, upper bound: 0.0005563
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033386, 0.0043263, 0.0034191, 0.0043338, -0.0004435, 0.0003543
1: 0.0018046, 0.0019473, 0.0018163, 0.0019484, -0.0000641, 0.0000512
2: 0.0119679, 0.0125140, 0.0119638, 0.0124695, -0.0001959, 0.0002452
3: -0.0023026, -0.0017379, -0.0023069, -0.0017839, -0.0002026, 0.0002536
4: -0.0021556, -0.0015442, -0.0021058, -0.0015396, -0.0002746, 0.0002193
5: 0.0055747, 0.0061533, 0.0055703, 0.0061061, -0.0002075, 0.0002598
6: -0.0001816, 0.0021141, -0.0001989, 0.0019270, -0.0008234, 0.0010309
7: -0.0054359, -0.0023094, -0.0051811, -0.0022859, -0.0014040, 0.0011215
8: 0.9853847, 0.9875870, 0.9855642, 0.9876037, -0.0009890, 0.0007900
9: -0.0046196, -0.0026205, -0.0046347, -0.0027834, -0.0007171, 0.0008978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005807, upper bound: 0.0005497
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0005497
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033618, 0.0043476, 0.0033923, 0.0043070, -0.0004367, 0.0004425
1: 0.0018080, 0.0019504, 0.0018124, 0.0019445, -0.0000631, 0.0000639
2: 0.0119562, 0.0125012, 0.0119786, 0.0124844, -0.0002447, 0.0002415
3: -0.0023148, -0.0017511, -0.0022916, -0.0017685, -0.0002530, 0.0002497
4: -0.0021413, -0.0015310, -0.0021224, -0.0015562, -0.0002703, 0.0002739
5: 0.0055622, 0.0061397, 0.0055860, 0.0061219, -0.0002592, 0.0002558
6: -0.0002311, 0.0020603, -0.0001367, 0.0019895, -0.0010285, 0.0010151
7: -0.0053626, -0.0022420, -0.0052662, -0.0023706, -0.0013825, 0.0014008
8: 0.9854363, 0.9876345, 0.9855043, 0.9875440, -0.0009738, 0.0009867
9: -0.0046627, -0.0026673, -0.0045805, -0.0027290, -0.0008957, 0.0008840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005502
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005502
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033386, 0.0043263, 0.0033936, 0.0042937, -0.0004627, 0.0004421
1: 0.0018046, 0.0019473, 0.0018126, 0.0019426, -0.0000668, 0.0000639
2: 0.0119679, 0.0125140, 0.0119860, 0.0124836, -0.0002444, 0.0002558
3: -0.0023026, -0.0017379, -0.0022840, -0.0017693, -0.0002528, 0.0002646
4: -0.0021556, -0.0015442, -0.0021216, -0.0015644, -0.0002864, 0.0002737
5: 0.0055747, 0.0061533, 0.0055938, 0.0061211, -0.0002590, 0.0002710
6: -0.0001816, 0.0021141, -0.0001058, 0.0019863, -0.0010275, 0.0010754
7: -0.0054359, -0.0023094, -0.0052618, -0.0024127, -0.0014646, 0.0013994
8: 0.9853847, 0.9875870, 0.9855073, 0.9875144, -0.0010317, 0.0009858
9: -0.0046196, -0.0026205, -0.0045536, -0.0027318, -0.0008948, 0.0009365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005404
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005404
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033935, 0.0043008, 0.0033797, 0.0042956, -0.0003952, 0.0004189
1: 0.0018126, 0.0019436, 0.0018106, 0.0019429, -0.0000571, 0.0000605
2: 0.0119821, 0.0124837, 0.0119850, 0.0124913, -0.0002316, 0.0002185
3: -0.0022880, -0.0017693, -0.0022851, -0.0017614, -0.0002395, 0.0002260
4: -0.0021216, -0.0015600, -0.0021302, -0.0015633, -0.0002446, 0.0002593
5: 0.0055897, 0.0061211, 0.0055927, 0.0061292, -0.0002454, 0.0002315
6: -0.0001222, 0.0019865, -0.0001101, 0.0020186, -0.0009737, 0.0009185
7: -0.0052622, -0.0023903, -0.0053059, -0.0024068, -0.0012509, 0.0013261
8: 0.9855070, 0.9875301, 0.9854763, 0.9875185, -0.0008811, 0.0009341
9: -0.0045680, -0.0027316, -0.0045574, -0.0027036, -0.0008479, 0.0007998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005381, upper bound: 0.0005501
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005430, upper bound: 0.0005501
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033717, 0.0042755, 0.0033811, 0.0042816, -0.0004232, 0.0004186
1: 0.0018094, 0.0019400, 0.0018108, 0.0019409, -0.0000611, 0.0000605
2: 0.0119961, 0.0124957, 0.0119927, 0.0124905, -0.0002314, 0.0002340
3: -0.0022736, -0.0017568, -0.0022771, -0.0017622, -0.0002394, 0.0002420
4: -0.0021351, -0.0015757, -0.0021293, -0.0015719, -0.0002620, 0.0002591
5: 0.0056045, 0.0061339, 0.0056009, 0.0061284, -0.0002452, 0.0002479
6: -0.0000634, 0.0020372, -0.0000777, 0.0020153, -0.0009729, 0.0009837
7: -0.0053312, -0.0024703, -0.0053014, -0.0024509, -0.0013397, 0.0013250
8: 0.9854584, 0.9874738, 0.9854794, 0.9874874, -0.0009437, 0.0009334
9: -0.0045168, -0.0026874, -0.0045292, -0.0027065, -0.0008473, 0.0008566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005381, upper bound: 0.0005411
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005430, upper bound: 0.0005411
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033935, 0.0043008, 0.0033499, 0.0042597, -0.0003555, 0.0004443
1: 0.0018126, 0.0019436, 0.0018063, 0.0019377, -0.0000514, 0.0000642
2: 0.0119821, 0.0124837, 0.0120048, 0.0125078, -0.0002457, 0.0001966
3: -0.0022880, -0.0017693, -0.0022646, -0.0017443, -0.0002541, 0.0002033
4: -0.0021216, -0.0015600, -0.0021486, -0.0015854, -0.0002201, 0.0002751
5: 0.0055897, 0.0061211, 0.0056137, 0.0061467, -0.0002603, 0.0002083
6: -0.0001222, 0.0019865, -0.0000268, 0.0020879, -0.0010328, 0.0008264
7: -0.0052622, -0.0023903, -0.0054002, -0.0025202, -0.0011254, 0.0014066
8: 0.9855070, 0.9875301, 0.9854099, 0.9874386, -0.0007928, 0.0009908
9: -0.0045680, -0.0027316, -0.0044849, -0.0026433, -0.0008994, 0.0007196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005501
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005501
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033717, 0.0042755, 0.0033513, 0.0042466, -0.0003868, 0.0004446
1: 0.0018094, 0.0019400, 0.0018065, 0.0019358, -0.0000559, 0.0000642
2: 0.0119961, 0.0124957, 0.0120120, 0.0125070, -0.0002458, 0.0002139
3: -0.0022736, -0.0017568, -0.0022571, -0.0017451, -0.0002542, 0.0002212
4: -0.0021351, -0.0015757, -0.0021478, -0.0015936, -0.0002395, 0.0002752
5: 0.0056045, 0.0061339, 0.0056214, 0.0061459, -0.0002604, 0.0002266
6: -0.0000634, 0.0020372, 0.0000037, 0.0020847, -0.0010333, 0.0008991
7: -0.0053312, -0.0024703, -0.0053959, -0.0025618, -0.0012245, 0.0014072
8: 0.9854584, 0.9874738, 0.9854129, 0.9874093, -0.0008626, 0.0009913
9: -0.0045168, -0.0026874, -0.0044583, -0.0026461, -0.0008998, 0.0007830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005411
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005411
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033344, 0.0043103, 0.0033797, 0.0042956, -0.0004666, 0.0004383
1: 0.0018040, 0.0019450, 0.0018106, 0.0019429, -0.0000674, 0.0000633
2: 0.0119768, 0.0125164, 0.0119850, 0.0124913, -0.0002423, 0.0002580
3: -0.0022935, -0.0017355, -0.0022851, -0.0017614, -0.0002506, 0.0002668
4: -0.0021582, -0.0015541, -0.0021302, -0.0015633, -0.0002888, 0.0002713
5: 0.0055841, 0.0061558, 0.0055927, 0.0061292, -0.0002567, 0.0002733
6: -0.0001444, 0.0021239, -0.0001101, 0.0020186, -0.0010187, 0.0010844
7: -0.0054493, -0.0023600, -0.0053059, -0.0024068, -0.0014769, 0.0013874
8: 0.9853753, 0.9875514, 0.9854763, 0.9875185, -0.0010404, 0.0009773
9: -0.0045873, -0.0026119, -0.0045574, -0.0027036, -0.0008871, 0.0009444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005501
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005501
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033127, 0.0042874, 0.0033811, 0.0042816, -0.0004974, 0.0004370
1: 0.0018009, 0.0019417, 0.0018108, 0.0019409, -0.0000719, 0.0000631
2: 0.0119895, 0.0125284, 0.0119927, 0.0124905, -0.0002416, 0.0002750
3: -0.0022804, -0.0017230, -0.0022771, -0.0017622, -0.0002499, 0.0002844
4: -0.0021717, -0.0015683, -0.0021293, -0.0015719, -0.0003079, 0.0002705
5: 0.0055975, 0.0061685, 0.0056009, 0.0061284, -0.0002560, 0.0002914
6: -0.0000910, 0.0021744, -0.0000777, 0.0020153, -0.0010157, 0.0011560
7: -0.0055180, -0.0024327, -0.0053014, -0.0024509, -0.0015744, 0.0013833
8: 0.9853269, 0.9875003, 0.9854794, 0.9874874, -0.0011090, 0.0009744
9: -0.0045408, -0.0025680, -0.0045292, -0.0027065, -0.0008845, 0.0010067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005411
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005411
time: 0.67 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.04 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005525, upper bound: 0.0005639
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005584, upper bound: 0.0005639
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005533, upper bound: 0.0005586
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005586
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005445, upper bound: 0.0005620
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005505, upper bound: 0.0005620
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0005548
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005505, upper bound: 0.0005548
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005759, upper bound: 0.0005638
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005772, upper bound: 0.0005638
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005765, upper bound: 0.0005584
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005774, upper bound: 0.0005584
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005683, upper bound: 0.0005618
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005690, upper bound: 0.0005618
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005686, upper bound: 0.0005548
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005692, upper bound: 0.0005548
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005491, upper bound: 0.0005640
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005552, upper bound: 0.0005640
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005497, upper bound: 0.0005561
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005553, upper bound: 0.0005561
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005358, upper bound: 0.0005535
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005536
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005358, upper bound: 0.0005430
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005430
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005728, upper bound: 0.0005628
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005740, upper bound: 0.0005628
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005736, upper bound: 0.0005558
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005744, upper bound: 0.0005558
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005531
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005614, upper bound: 0.0005531
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005609, upper bound: 0.0005430
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005615, upper bound: 0.0005430
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005490, upper bound: 0.0005567
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005548, upper bound: 0.0005567
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005495, upper bound: 0.0005505
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005548, upper bound: 0.0005505
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005567
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005498, upper bound: 0.0005567
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005442, upper bound: 0.0005505
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005499, upper bound: 0.0005505
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005732, upper bound: 0.0005567
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005742, upper bound: 0.0005567
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005738, upper bound: 0.0005505
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005744, upper bound: 0.0005505
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005679, upper bound: 0.0005567
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005688, upper bound: 0.0005567
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005684, upper bound: 0.0005505
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005692, upper bound: 0.0005505
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005430
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005477, upper bound: 0.0005430
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005313
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005477, upper bound: 0.0005313
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005355, upper bound: 0.0005430
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005410, upper bound: 0.0005430
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005355, upper bound: 0.0005313
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005410, upper bound: 0.0005313
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005672, upper bound: 0.0005430
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005678, upper bound: 0.0005430
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005676, upper bound: 0.0005313
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005680, upper bound: 0.0005313
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005430
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005612, upper bound: 0.0005430
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005609, upper bound: 0.0005313
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005614, upper bound: 0.0005313
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005518, upper bound: 0.0005847
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005582, upper bound: 0.0005847
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005524, upper bound: 0.0005774
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005584, upper bound: 0.0005774
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005445, upper bound: 0.0005833
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005505, upper bound: 0.0005833
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0005744
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005505, upper bound: 0.0005744
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005779, upper bound: 0.0005657
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005797, upper bound: 0.0005658
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005786, upper bound: 0.0005608
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005799, upper bound: 0.0005609
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005722, upper bound: 0.0005652
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005738, upper bound: 0.0005652
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005727, upper bound: 0.0005601
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005747, upper bound: 0.0005601
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005874
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005552, upper bound: 0.0005874
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005493, upper bound: 0.0005814
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005552, upper bound: 0.0005814
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005358, upper bound: 0.0005797
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005797
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005358, upper bound: 0.0005693
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005693
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005773, upper bound: 0.0005684
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005789, upper bound: 0.0005684
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005779, upper bound: 0.0005631
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005792, upper bound: 0.0005631
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005698, upper bound: 0.0005664
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005714, upper bound: 0.0005664
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005701, upper bound: 0.0005592
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005716, upper bound: 0.0005592
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005488, upper bound: 0.0005773
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005547, upper bound: 0.0005773
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005491, upper bound: 0.0005692
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005548, upper bound: 0.0005692
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005773
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005498, upper bound: 0.0005773
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005442, upper bound: 0.0005692
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005499, upper bound: 0.0005692
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005775, upper bound: 0.0005607
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005790, upper bound: 0.0005607
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005780, upper bound: 0.0005562
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005792, upper bound: 0.0005562
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005722, upper bound: 0.0005607
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005738, upper bound: 0.0005607
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005727, upper bound: 0.0005561
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005562
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005415, upper bound: 0.0005732
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005477, upper bound: 0.0005732
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005415, upper bound: 0.0005578
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005477, upper bound: 0.0005578
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005355, upper bound: 0.0005732
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005410, upper bound: 0.0005732
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005355, upper bound: 0.0005578
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005410, upper bound: 0.0005578
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005580
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005771, upper bound: 0.0005580
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005762, upper bound: 0.0005506
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005773, upper bound: 0.0005506
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005695, upper bound: 0.0005580
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005713, upper bound: 0.0005580
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005700, upper bound: 0.0005506
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005715, upper bound: 0.0005506
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005517, upper bound: 0.0005619
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005558, upper bound: 0.0005619
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005519, upper bound: 0.0005553
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005561, upper bound: 0.0005553
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005564
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005564
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005477
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005477
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005802, upper bound: 0.0005618
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005812, upper bound: 0.0005618
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005807, upper bound: 0.0005553
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005814, upper bound: 0.0005552
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005562
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005562
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005477
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005477
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005517, upper bound: 0.0005565
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005558, upper bound: 0.0005565
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005519, upper bound: 0.0005499
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005561, upper bound: 0.0005499
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005505
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005505
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005404
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005404
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005802, upper bound: 0.0005563
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005812, upper bound: 0.0005563
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005807, upper bound: 0.0005497
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005813, upper bound: 0.0005497
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005502
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005502
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005404
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005578, upper bound: 0.0005404
IS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005381, upper bound: 0.0005501
IS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005430, upper bound: 0.0005501
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005381, upper bound: 0.0005411
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005430, upper bound: 0.0005411
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005501
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005501
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005260, upper bound: 0.0005411
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005313, upper bound: 0.0005411
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005501
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005501
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005411
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.04
Output dim: 8, lower bound: -0.0005693, upper bound: 0.0005411
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005647
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005568
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005555
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005463
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005555
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005463
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005554
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005463
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005554
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005757, upper bound: 0.0005463
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005927
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005821
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005927
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005821
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005778
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005708
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005778
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005707
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005896
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005794
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005896
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005794
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005730
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005660
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005730
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005660
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005858
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005767
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005858
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005767
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005733
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005667
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005733
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005667
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005822
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005716
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005822
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005492, upper bound: 0.0005716
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005650
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005579
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005860, upper bound: 0.0005650
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.0005861, upper bound: 0.0005579

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.73 + 598.52 = 601.24 seconds
