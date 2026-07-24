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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0032617, 0.0042679, 0.0032617, 0.0042679, -0.0005474, 0.0005474)
1: (0.0017935, 0.0019389, 0.0017935, 0.0019389, -0.0000791, 0.0000791)
2: (0.0120002, 0.0125565, 0.0120002, 0.0125565, -0.0003026, 0.0003026)
3: (-0.0022692, -0.0016939, -0.0022692, -0.0016939, -0.0003130, 0.0003130)
4: (-0.0022032, -0.0015804, -0.0022032, -0.0015804, -0.0003388, 0.0003388)
5: (0.0056089, 0.0061983, 0.0056089, 0.0061983, -0.0003207, 0.0003207)
6: (-0.0000458, 0.0022928, -0.0000458, 0.0022928, -0.0012723, 0.0012723)
7: (-0.0056794, -0.0024944, -0.0056794, -0.0024944, -0.0017327, 0.0017327)
8: (0.9852133, 0.9874567, 0.9852133, 0.9874567, -0.0012206, 0.0012206)
9: (-0.0045014, -0.0024648, -0.0045014, -0.0024648, -0.0011079, 0.0011079)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.37 + 1.41 = 2.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0007225, upper bound: 0.0007226

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006566, upper bound: 0.0006733
time: 0.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006733, upper bound: 0.0006733
time: 0.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.31 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.31
Output dim: 8, lower bound: -0.0006566, upper bound: 0.0006733
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.31
Output dim: 8, lower bound: -0.0006733, upper bound: 0.0006733

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0033127, 0.0042672, 0.0032779, 0.0042677, -0.0004797, 0.0005242
1: 0.0018009, 0.0019388, 0.0017959, 0.0019389, -0.0000693, 0.0000757
2: 0.0120007, 0.0125283, 0.0120004, 0.0125476, -0.0002898, 0.0002652
3: -0.0022688, -0.0017231, -0.0022691, -0.0017032, -0.0002997, 0.0002743
4: -0.0021716, -0.0015808, -0.0021932, -0.0015805, -0.0002969, 0.0003245
5: 0.0056093, 0.0061685, 0.0056090, 0.0061888, -0.0003071, 0.0002810
6: -0.0000441, 0.0021743, -0.0000453, 0.0022552, -0.0012183, 0.0011149
7: -0.0055179, -0.0024967, -0.0056281, -0.0024950, -0.0015183, 0.0016592
8: 0.9853269, 0.9874551, 0.9852493, 0.9874564, -0.0010695, 0.0011688
9: -0.0044999, -0.0025681, -0.0045010, -0.0024976, -0.0010610, 0.0009709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006144, upper bound: 0.0006298
time: 0.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006164, upper bound: 0.0006343
time: 0.57 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0033353, 0.0043494, 0.0032997, 0.0042676, -0.0004825, 0.0006671
1: 0.0018042, 0.0019507, 0.0017990, 0.0019389, -0.0000697, 0.0000964
2: 0.0119552, 0.0125159, 0.0120004, 0.0125356, -0.0003688, 0.0002668
3: -0.0023158, -0.0017360, -0.0022691, -0.0017156, -0.0003814, 0.0002759
4: -0.0021577, -0.0015300, -0.0021797, -0.0015805, -0.0002987, 0.0004129
5: 0.0055612, 0.0061553, 0.0056091, 0.0061761, -0.0003908, 0.0002827
6: -0.0002351, 0.0021219, -0.0000452, 0.0022046, -0.0015504, 0.0011215
7: -0.0054466, -0.0022365, -0.0055592, -0.0024952, -0.0015274, 0.0021115
8: 0.9853772, 0.9876384, 0.9852979, 0.9874562, -0.0010760, 0.0014874
9: -0.0046663, -0.0026137, -0.0045009, -0.0025416, -0.0013502, 0.0009767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006318, upper bound: 0.0006298
time: 0.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006342, upper bound: 0.0006343
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.56 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 8, lower bound: -0.0006144, upper bound: 0.0006298
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 8, lower bound: -0.0006164, upper bound: 0.0006343
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 8, lower bound: -0.0006318, upper bound: 0.0006298
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 8, lower bound: -0.0006342, upper bound: 0.0006343

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033413, 0.0042670, 0.0033421, 0.0043029, -0.0004065, 0.0004267
1: 0.0018050, 0.0019388, 0.0018051, 0.0019439, -0.0000587, 0.0000616
2: 0.0120007, 0.0125126, 0.0119809, 0.0125121, -0.0002359, 0.0002247
3: -0.0022687, -0.0017394, -0.0022893, -0.0017398, -0.0002440, 0.0002324
4: -0.0021540, -0.0015809, -0.0021535, -0.0015587, -0.0002516, 0.0002641
5: 0.0056094, 0.0061517, 0.0055884, 0.0061513, -0.0002499, 0.0002381
6: -0.0000438, 0.0021080, -0.0001272, 0.0021061, -0.0009917, 0.0009448
7: -0.0054276, -0.0024971, -0.0054250, -0.0023835, -0.0012867, 0.0013506
8: 0.9853905, 0.9874549, 0.9853923, 0.9875348, -0.0009064, 0.0009514
9: -0.0044996, -0.0026258, -0.0045722, -0.0026275, -0.0008636, 0.0008227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005829, upper bound: 0.0005935
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005791, upper bound: 0.0005935
time: 0.53 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033193, 0.0042672, 0.0033040, 0.0042676, -0.0004780, 0.0004024
1: 0.0018018, 0.0019388, 0.0017996, 0.0019388, -0.0000691, 0.0000581
2: 0.0120007, 0.0125247, 0.0120004, 0.0125331, -0.0002225, 0.0002643
3: -0.0022688, -0.0017268, -0.0022691, -0.0017181, -0.0002301, 0.0002733
4: -0.0021676, -0.0015808, -0.0021770, -0.0015806, -0.0002959, 0.0002491
5: 0.0056094, 0.0061646, 0.0056091, 0.0061735, -0.0002357, 0.0002800
6: -0.0000440, 0.0021590, -0.0000451, 0.0021945, -0.0009352, 0.0011111
7: -0.0054971, -0.0024967, -0.0055454, -0.0024953, -0.0015132, 0.0012736
8: 0.9853415, 0.9874551, 0.9853076, 0.9874561, -0.0010659, 0.0008972
9: -0.0044999, -0.0025814, -0.0045008, -0.0025505, -0.0008144, 0.0009676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005844, upper bound: 0.0006003
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005805, upper bound: 0.0006004
time: 0.58 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033651, 0.0043492, 0.0033672, 0.0043028, -0.0004097, 0.0005829
1: 0.0018085, 0.0019506, 0.0018088, 0.0019439, -0.0000592, 0.0000842
2: 0.0119553, 0.0124994, 0.0119809, 0.0124982, -0.0003223, 0.0002265
3: -0.0023157, -0.0017530, -0.0022892, -0.0017542, -0.0003333, 0.0002343
4: -0.0021392, -0.0015300, -0.0021379, -0.0015587, -0.0002536, 0.0003608
5: 0.0055613, 0.0061378, 0.0055885, 0.0061365, -0.0003415, 0.0002400
6: -0.0002348, 0.0020525, -0.0001270, 0.0020476, -0.0013549, 0.0009522
7: -0.0053520, -0.0022369, -0.0053454, -0.0023838, -0.0012968, 0.0018452
8: 0.9854438, 0.9876382, 0.9854485, 0.9875347, -0.0009135, 0.0012998
9: -0.0046660, -0.0026741, -0.0045721, -0.0026784, -0.0011799, 0.0008292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006001, upper bound: 0.0005935
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005982, upper bound: 0.0005935
time: 0.56 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033416, 0.0043493, 0.0033244, 0.0042676, -0.0004808, 0.0005950
1: 0.0018051, 0.0019507, 0.0018026, 0.0019388, -0.0000695, 0.0000860
2: 0.0119552, 0.0125124, 0.0120004, 0.0125219, -0.0003290, 0.0002658
3: -0.0023158, -0.0017396, -0.0022690, -0.0017298, -0.0003402, 0.0002749
4: -0.0021538, -0.0015300, -0.0021644, -0.0015806, -0.0002976, 0.0003683
5: 0.0055612, 0.0061515, 0.0056091, 0.0061616, -0.0003486, 0.0002816
6: -0.0002351, 0.0021071, -0.0000450, 0.0021471, -0.0013830, 0.0011175
7: -0.0054264, -0.0022366, -0.0054808, -0.0024954, -0.0015219, 0.0018836
8: 0.9853914, 0.9876384, 0.9853531, 0.9874560, -0.0010721, 0.0013268
9: -0.0046662, -0.0026265, -0.0045007, -0.0025918, -0.0012044, 0.0009732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0006021
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0006004
time: 0.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 8, lower bound: -0.0005829, upper bound: 0.0005935
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 8, lower bound: -0.0005791, upper bound: 0.0005935
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 8, lower bound: -0.0005844, upper bound: 0.0006003
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 8, lower bound: -0.0005805, upper bound: 0.0006004
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 8, lower bound: -0.0006001, upper bound: 0.0005935
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 8, lower bound: -0.0005982, upper bound: 0.0005935
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0006021
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0006004

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033420, 0.0042600, 0.0033446, 0.0042818, -0.0003713, 0.0004143
1: 0.0018051, 0.0019377, 0.0018055, 0.0019409, -0.0000536, 0.0000599
2: 0.0120046, 0.0125122, 0.0119926, 0.0125107, -0.0002291, 0.0002053
3: -0.0022647, -0.0017398, -0.0022772, -0.0017413, -0.0002369, 0.0002123
4: -0.0021535, -0.0015853, -0.0021519, -0.0015718, -0.0002299, 0.0002565
5: 0.0056136, 0.0061513, 0.0056008, 0.0061498, -0.0002427, 0.0002175
6: -0.0000273, 0.0021063, -0.0000781, 0.0021002, -0.0009630, 0.0008631
7: -0.0054253, -0.0025195, -0.0054169, -0.0024503, -0.0011754, 0.0013115
8: 0.9853922, 0.9874391, 0.9853981, 0.9874879, -0.0008280, 0.0009239
9: -0.0044853, -0.0026272, -0.0045295, -0.0026326, -0.0008386, 0.0007516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005829, upper bound: 0.0005755
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005829, upper bound: 0.0005935
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033430, 0.0042438, 0.0033247, 0.0042478, -0.0003657, 0.0004877
1: 0.0018053, 0.0019354, 0.0018026, 0.0019360, -0.0000528, 0.0000705
2: 0.0120136, 0.0125116, 0.0120114, 0.0125217, -0.0002696, 0.0002022
3: -0.0022555, -0.0017404, -0.0022578, -0.0017299, -0.0002789, 0.0002091
4: -0.0021529, -0.0015953, -0.0021642, -0.0015928, -0.0002264, 0.0003019
5: 0.0056230, 0.0061507, 0.0056207, 0.0061614, -0.0002857, 0.0002142
6: 0.0000101, 0.0021040, 0.0000009, 0.0021465, -0.0011335, 0.0008500
7: -0.0054222, -0.0025705, -0.0054801, -0.0025579, -0.0011577, 0.0015437
8: 0.9853943, 0.9874032, 0.9853535, 0.9874120, -0.0008155, 0.0010874
9: -0.0044527, -0.0026292, -0.0044607, -0.0025923, -0.0009871, 0.0007403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005791, upper bound: 0.0005755
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005791, upper bound: 0.0005935
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033200, 0.0042601, 0.0033065, 0.0042452, -0.0004459, 0.0003881
1: 0.0018019, 0.0019378, 0.0018000, 0.0019356, -0.0000644, 0.0000561
2: 0.0120046, 0.0125243, 0.0120128, 0.0125318, -0.0002146, 0.0002465
3: -0.0022648, -0.0017272, -0.0022562, -0.0017195, -0.0002219, 0.0002550
4: -0.0021671, -0.0015852, -0.0021755, -0.0015944, -0.0002760, 0.0002402
5: 0.0056135, 0.0061642, 0.0056222, 0.0061721, -0.0002273, 0.0002612
6: -0.0000276, 0.0021573, 0.0000070, 0.0021887, -0.0009020, 0.0010364
7: -0.0054948, -0.0025191, -0.0055376, -0.0025663, -0.0014115, 0.0012284
8: 0.9853432, 0.9874394, 0.9853131, 0.9874061, -0.0009943, 0.0008653
9: -0.0044856, -0.0025828, -0.0044554, -0.0025555, -0.0007855, 0.0009026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005844, upper bound: 0.0005786
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005844, upper bound: 0.0006004
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033210, 0.0042440, 0.0032827, 0.0042161, -0.0004407, 0.0004701
1: 0.0018021, 0.0019354, 0.0017965, 0.0019314, -0.0000637, 0.0000679
2: 0.0120135, 0.0125237, 0.0120289, 0.0125450, -0.0002599, 0.0002437
3: -0.0022556, -0.0017278, -0.0022396, -0.0017059, -0.0002688, 0.0002520
4: -0.0021665, -0.0015952, -0.0021903, -0.0016124, -0.0002728, 0.0002910
5: 0.0056229, 0.0061636, 0.0056392, 0.0061861, -0.0002754, 0.0002582
6: 0.0000099, 0.0021550, 0.0000745, 0.0022442, -0.0010927, 0.0010244
7: -0.0054916, -0.0025701, -0.0056131, -0.0026582, -0.0013951, 0.0014881
8: 0.9853455, 0.9874034, 0.9852600, 0.9873414, -0.0009828, 0.0010483
9: -0.0044529, -0.0025849, -0.0043966, -0.0025072, -0.0009515, 0.0008921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005806, upper bound: 0.0005786
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005806, upper bound: 0.0006004
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033658, 0.0043421, 0.0033698, 0.0042817, -0.0003746, 0.0005721
1: 0.0018086, 0.0019496, 0.0018091, 0.0019409, -0.0000541, 0.0000827
2: 0.0119592, 0.0124990, 0.0119926, 0.0124968, -0.0003163, 0.0002071
3: -0.0023116, -0.0017534, -0.0022772, -0.0017557, -0.0003271, 0.0002142
4: -0.0021388, -0.0015345, -0.0021363, -0.0015718, -0.0002319, 0.0003541
5: 0.0055655, 0.0061373, 0.0056008, 0.0061350, -0.0003351, 0.0002194
6: -0.0002182, 0.0020509, -0.0000780, 0.0020416, -0.0013297, 0.0008706
7: -0.0053498, -0.0022596, -0.0053372, -0.0024505, -0.0011856, 0.0018110
8: 0.9854453, 0.9876221, 0.9854543, 0.9874877, -0.0008352, 0.0012757
9: -0.0046515, -0.0026755, -0.0045294, -0.0026836, -0.0011580, 0.0007581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006001, upper bound: 0.0005755
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006001, upper bound: 0.0005755
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033670, 0.0043262, 0.0033484, 0.0042477, -0.0003695, 0.0006194
1: 0.0018087, 0.0019473, 0.0018060, 0.0019360, -0.0000534, 0.0000895
2: 0.0119680, 0.0124983, 0.0120114, 0.0125086, -0.0003425, 0.0002043
3: -0.0023026, -0.0017541, -0.0022577, -0.0017434, -0.0003542, 0.0002113
4: -0.0021380, -0.0015443, -0.0021496, -0.0015929, -0.0002287, 0.0003834
5: 0.0055748, 0.0061367, 0.0056207, 0.0061476, -0.0003628, 0.0002165
6: -0.0001813, 0.0020481, 0.0000011, 0.0020915, -0.0014397, 0.0008589
7: -0.0053461, -0.0023098, -0.0054051, -0.0025582, -0.0011697, 0.0019607
8: 0.9854480, 0.9875867, 0.9854064, 0.9874119, -0.0008239, 0.0013812
9: -0.0046194, -0.0026779, -0.0044606, -0.0026402, -0.0012537, 0.0007479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005982, upper bound: 0.0005755
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005982, upper bound: 0.0005755
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0043262, 0.0033252, 0.0042605, -0.0004685, 0.0005662
1: 0.0018054, 0.0019473, 0.0018027, 0.0019378, -0.0000677, 0.0000818
2: 0.0119680, 0.0125111, 0.0120043, 0.0125215, -0.0003130, 0.0002590
3: -0.0023026, -0.0017409, -0.0022650, -0.0017302, -0.0003238, 0.0002679
4: -0.0021523, -0.0015443, -0.0021639, -0.0015850, -0.0002900, 0.0003505
5: 0.0055748, 0.0061501, 0.0056133, 0.0061612, -0.0003317, 0.0002744
6: -0.0001812, 0.0021016, -0.0000285, 0.0021454, -0.0013160, 0.0010888
7: -0.0054189, -0.0023099, -0.0054785, -0.0025178, -0.0014829, 0.0017923
8: 0.9853967, 0.9875867, 0.9853547, 0.9874403, -0.0010446, 0.0012625
9: -0.0046193, -0.0026313, -0.0044864, -0.0025932, -0.0011461, 0.0009482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0005817
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0005817
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033218, 0.0042973, 0.0033262, 0.0042444, -0.0005353, 0.0005614
1: 0.0018022, 0.0019431, 0.0018028, 0.0019355, -0.0000773, 0.0000811
2: 0.0119840, 0.0125233, 0.0120133, 0.0125209, -0.0003104, 0.0002960
3: -0.0022860, -0.0017282, -0.0022558, -0.0017308, -0.0003210, 0.0003061
4: -0.0021660, -0.0015622, -0.0021633, -0.0015949, -0.0003314, 0.0003475
5: 0.0055917, 0.0061632, 0.0056227, 0.0061606, -0.0003289, 0.0003136
6: -0.0001141, 0.0021532, 0.0000089, 0.0021430, -0.0013049, 0.0012442
7: -0.0054892, -0.0024013, -0.0054752, -0.0025688, -0.0016946, 0.0017771
8: 0.9853472, 0.9875223, 0.9853569, 0.9874043, -0.0011937, 0.0012518
9: -0.0045609, -0.0025864, -0.0044538, -0.0025953, -0.0011363, 0.0010835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0005786
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0005786
time: 0.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005829, upper bound: 0.0005755
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005829, upper bound: 0.0005935
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005791, upper bound: 0.0005755
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005791, upper bound: 0.0005935
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005844, upper bound: 0.0005786
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005844, upper bound: 0.0006004
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005806, upper bound: 0.0005786
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005806, upper bound: 0.0006004
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0006001, upper bound: 0.0005755
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0006001, upper bound: 0.0005755
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005982, upper bound: 0.0005755
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0005982, upper bound: 0.0005755
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0005817
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0005817
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0005786
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.63
Output dim: 8, lower bound: -0.0006004, upper bound: 0.0005786

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033420, 0.0042600, 0.0033775, 0.0042812, -0.0003711, 0.0003648
1: 0.0018051, 0.0019377, 0.0018103, 0.0019408, -0.0000536, 0.0000527
2: 0.0120046, 0.0125122, 0.0119929, 0.0124925, -0.0002017, 0.0002052
3: -0.0022647, -0.0017398, -0.0022769, -0.0017601, -0.0002086, 0.0002122
4: -0.0021535, -0.0015853, -0.0021315, -0.0015721, -0.0002297, 0.0002258
5: 0.0056136, 0.0061513, 0.0056011, 0.0061305, -0.0002137, 0.0002174
6: -0.0000273, 0.0021063, -0.0000768, 0.0020237, -0.0008479, 0.0008625
7: -0.0054253, -0.0025195, -0.0053129, -0.0024521, -0.0011746, 0.0011548
8: 0.9853922, 0.9874391, 0.9854713, 0.9874865, -0.0008274, 0.0008135
9: -0.0044853, -0.0026272, -0.0045284, -0.0026992, -0.0007384, 0.0007511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005413
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005332
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033420, 0.0042600, 0.0034066, 0.0043551, -0.0005502, 0.0004318
1: 0.0018051, 0.0019377, 0.0018145, 0.0019515, -0.0000795, 0.0000624
2: 0.0120046, 0.0125122, 0.0119521, 0.0124764, -0.0002388, 0.0003042
3: -0.0022647, -0.0017398, -0.0023191, -0.0017768, -0.0002469, 0.0003146
4: -0.0021535, -0.0015853, -0.0021135, -0.0015264, -0.0003406, 0.0002673
5: 0.0056136, 0.0061513, 0.0055579, 0.0061135, -0.0002530, 0.0003223
6: -0.0000273, 0.0021063, -0.0002484, 0.0019561, -0.0010037, 0.0012788
7: -0.0054253, -0.0025195, -0.0052207, -0.0022185, -0.0017416, 0.0013670
8: 0.9853922, 0.9874391, 0.9855363, 0.9876511, -0.0012268, 0.0009629
9: -0.0044853, -0.0026272, -0.0046778, -0.0027581, -0.0008741, 0.0011136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005568, upper bound: 0.0005397
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005387
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033430, 0.0042438, 0.0033566, 0.0042472, -0.0003654, 0.0004434
1: 0.0018053, 0.0019354, 0.0018072, 0.0019359, -0.0000528, 0.0000641
2: 0.0120136, 0.0125116, 0.0120117, 0.0125041, -0.0002452, 0.0002020
3: -0.0022555, -0.0017404, -0.0022574, -0.0017481, -0.0002536, 0.0002090
4: -0.0021529, -0.0015953, -0.0021445, -0.0015932, -0.0002262, 0.0002745
5: 0.0056230, 0.0061507, 0.0056210, 0.0061428, -0.0002598, 0.0002141
6: 0.0000101, 0.0021040, 0.0000022, 0.0020724, -0.0010307, 0.0008494
7: -0.0054222, -0.0025705, -0.0053792, -0.0025597, -0.0011568, 0.0014037
8: 0.9853943, 0.9874032, 0.9854247, 0.9874107, -0.0008149, 0.0009888
9: -0.0044527, -0.0026292, -0.0044596, -0.0026567, -0.0008976, 0.0007397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005452, upper bound: 0.0005413
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005448, upper bound: 0.0005332
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033430, 0.0042438, 0.0033859, 0.0043220, -0.0005442, 0.0004716
1: 0.0018053, 0.0019354, 0.0018115, 0.0019467, -0.0000786, 0.0000681
2: 0.0120136, 0.0125116, 0.0119703, 0.0124879, -0.0002607, 0.0003009
3: -0.0022555, -0.0017404, -0.0023002, -0.0017649, -0.0002696, 0.0003112
4: -0.0021529, -0.0015953, -0.0021263, -0.0015469, -0.0003369, 0.0002919
5: 0.0056230, 0.0061507, 0.0055772, 0.0061256, -0.0002762, 0.0003188
6: 0.0000101, 0.0021040, -0.0001716, 0.0020042, -0.0010960, 0.0012648
7: -0.0054222, -0.0025705, -0.0052862, -0.0023230, -0.0017226, 0.0014927
8: 0.9853943, 0.9874032, 0.9854901, 0.9875774, -0.0012134, 0.0010515
9: -0.0044527, -0.0026292, -0.0046109, -0.0027162, -0.0009545, 0.0011015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005520, upper bound: 0.0005397
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005448, upper bound: 0.0005387
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033200, 0.0042601, 0.0033420, 0.0042447, -0.0004456, 0.0003393
1: 0.0018019, 0.0019378, 0.0018051, 0.0019355, -0.0000644, 0.0000490
2: 0.0120046, 0.0125243, 0.0120131, 0.0125122, -0.0001876, 0.0002464
3: -0.0022648, -0.0017272, -0.0022559, -0.0017398, -0.0001940, 0.0002548
4: -0.0021671, -0.0015852, -0.0021535, -0.0015948, -0.0002758, 0.0002100
5: 0.0056135, 0.0061642, 0.0056225, 0.0061513, -0.0001988, 0.0002610
6: -0.0000276, 0.0021573, 0.0000083, 0.0021063, -0.0007887, 0.0010357
7: -0.0054948, -0.0025191, -0.0054253, -0.0025680, -0.0014105, 0.0010741
8: 0.9853432, 0.9874394, 0.9853922, 0.9874050, -0.0009936, 0.0007566
9: -0.0044856, -0.0025828, -0.0044543, -0.0026272, -0.0006868, 0.0009019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005771
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005764
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033200, 0.0042601, 0.0033635, 0.0043261, -0.0005852, 0.0004054
1: 0.0018019, 0.0019378, 0.0018082, 0.0019473, -0.0000846, 0.0000586
2: 0.0120046, 0.0125243, 0.0119681, 0.0125003, -0.0002241, 0.0003236
3: -0.0022648, -0.0017272, -0.0023025, -0.0017521, -0.0002318, 0.0003346
4: -0.0021671, -0.0015852, -0.0021402, -0.0015443, -0.0003623, 0.0002509
5: 0.0056135, 0.0061642, 0.0055748, 0.0061387, -0.0002375, 0.0003428
6: -0.0000276, 0.0021573, -0.0001811, 0.0020563, -0.0009422, 0.0013603
7: -0.0054948, -0.0025191, -0.0053572, -0.0023101, -0.0018526, 0.0012832
8: 0.9853432, 0.9874394, 0.9854401, 0.9875866, -0.0013050, 0.0009039
9: -0.0044856, -0.0025828, -0.0046192, -0.0026708, -0.0008205, 0.0011846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005982
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005954
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033210, 0.0042440, 0.0033181, 0.0042156, -0.0004404, 0.0004256
1: 0.0018021, 0.0019354, 0.0018017, 0.0019313, -0.0000636, 0.0000615
2: 0.0120135, 0.0125237, 0.0120292, 0.0125254, -0.0002353, 0.0002435
3: -0.0022556, -0.0017278, -0.0022393, -0.0017261, -0.0002434, 0.0002518
4: -0.0021665, -0.0015952, -0.0021683, -0.0016127, -0.0002726, 0.0002635
5: 0.0056229, 0.0061636, 0.0056396, 0.0061653, -0.0002493, 0.0002580
6: 0.0000099, 0.0021550, 0.0000758, 0.0021618, -0.0009892, 0.0010237
7: -0.0054916, -0.0025701, -0.0055010, -0.0026599, -0.0013942, 0.0013472
8: 0.9853455, 0.9874034, 0.9853389, 0.9873402, -0.0009821, 0.0009490
9: -0.0044529, -0.0025849, -0.0043955, -0.0025789, -0.0008615, 0.0008915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005778, upper bound: 0.0005771
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005778, upper bound: 0.0005764
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033210, 0.0042440, 0.0033403, 0.0042972, -0.0005792, 0.0004545
1: 0.0018021, 0.0019354, 0.0018049, 0.0019431, -0.0000837, 0.0000657
2: 0.0120135, 0.0125237, 0.0119840, 0.0125131, -0.0002513, 0.0003202
3: -0.0022556, -0.0017278, -0.0022860, -0.0017389, -0.0002599, 0.0003312
4: -0.0021665, -0.0015952, -0.0021545, -0.0015622, -0.0003585, 0.0002814
5: 0.0056229, 0.0061636, 0.0055917, 0.0061523, -0.0002663, 0.0003393
6: 0.0000099, 0.0021550, -0.0001140, 0.0021101, -0.0010564, 0.0013462
7: -0.0054916, -0.0025701, -0.0054305, -0.0024015, -0.0018334, 0.0014388
8: 0.9853455, 0.9874034, 0.9853885, 0.9875222, -0.0012915, 0.0010135
9: -0.0044529, -0.0025849, -0.0045608, -0.0026239, -0.0009200, 0.0011723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005778, upper bound: 0.0005982
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005778, upper bound: 0.0005955
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033658, 0.0043421, 0.0033775, 0.0042812, -0.0004266, 0.0005027
1: 0.0018086, 0.0019496, 0.0018103, 0.0019408, -0.0000616, 0.0000726
2: 0.0119592, 0.0124990, 0.0119929, 0.0124925, -0.0002779, 0.0002359
3: -0.0023116, -0.0017534, -0.0022769, -0.0017601, -0.0002874, 0.0002440
4: -0.0021388, -0.0015345, -0.0021315, -0.0015721, -0.0002641, 0.0003112
5: 0.0055655, 0.0061373, 0.0056011, 0.0061305, -0.0002945, 0.0002499
6: -0.0002182, 0.0020509, -0.0000768, 0.0020237, -0.0011684, 0.0009916
7: -0.0053498, -0.0022596, -0.0053129, -0.0024521, -0.0013505, 0.0015912
8: 0.9854453, 0.9876221, 0.9854713, 0.9874865, -0.0009513, 0.0011209
9: -0.0046515, -0.0026755, -0.0045284, -0.0026992, -0.0010175, 0.0008635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005413
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005332
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033658, 0.0043421, 0.0034066, 0.0043551, -0.0003799, 0.0003733
1: 0.0018086, 0.0019496, 0.0018145, 0.0019515, -0.0000549, 0.0000539
2: 0.0119592, 0.0124990, 0.0119521, 0.0124764, -0.0002064, 0.0002100
3: -0.0023116, -0.0017534, -0.0023191, -0.0017768, -0.0002135, 0.0002172
4: -0.0021388, -0.0015345, -0.0021135, -0.0015264, -0.0002352, 0.0002311
5: 0.0055655, 0.0061373, 0.0055579, 0.0061135, -0.0002187, 0.0002225
6: -0.0002182, 0.0020509, -0.0002484, 0.0019561, -0.0008677, 0.0008830
7: -0.0053498, -0.0022596, -0.0052207, -0.0022185, -0.0012026, 0.0011817
8: 0.9854453, 0.9876221, 0.9855363, 0.9876511, -0.0008471, 0.0008324
9: -0.0046515, -0.0026755, -0.0046778, -0.0027581, -0.0007556, 0.0007690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005413
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005332
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033670, 0.0043262, 0.0033566, 0.0042472, -0.0004209, 0.0005754
1: 0.0018087, 0.0019473, 0.0018072, 0.0019359, -0.0000608, 0.0000831
2: 0.0119680, 0.0124983, 0.0120117, 0.0125041, -0.0003181, 0.0002327
3: -0.0023026, -0.0017541, -0.0022574, -0.0017481, -0.0003290, 0.0002407
4: -0.0021380, -0.0015443, -0.0021445, -0.0015932, -0.0002606, 0.0003562
5: 0.0055748, 0.0061367, 0.0056210, 0.0061428, -0.0003371, 0.0002466
6: -0.0001813, 0.0020481, 0.0000022, 0.0020724, -0.0013375, 0.0009783
7: -0.0053461, -0.0023098, -0.0053792, -0.0025597, -0.0013324, 0.0018215
8: 0.9854480, 0.9875867, 0.9854247, 0.9874107, -0.0009386, 0.0012831
9: -0.0046194, -0.0026779, -0.0044596, -0.0026567, -0.0011647, 0.0008520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005507, upper bound: 0.0005413
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005499, upper bound: 0.0005332
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033670, 0.0043262, 0.0033859, 0.0043220, -0.0003751, 0.0004518
1: 0.0018087, 0.0019473, 0.0018115, 0.0019467, -0.0000542, 0.0000653
2: 0.0119680, 0.0124983, 0.0119703, 0.0124879, -0.0002498, 0.0002074
3: -0.0023026, -0.0017541, -0.0023002, -0.0017649, -0.0002583, 0.0002145
4: -0.0021380, -0.0015443, -0.0021263, -0.0015469, -0.0002322, 0.0002797
5: 0.0055748, 0.0061367, 0.0055772, 0.0061256, -0.0002647, 0.0002197
6: -0.0001813, 0.0020481, -0.0001716, 0.0020042, -0.0010501, 0.0008717
7: -0.0053461, -0.0023098, -0.0052862, -0.0023230, -0.0011872, 0.0014301
8: 0.9854480, 0.9875867, 0.9854901, 0.9875774, -0.0008363, 0.0010074
9: -0.0046194, -0.0026779, -0.0046109, -0.0027162, -0.0009145, 0.0007591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005507, upper bound: 0.0005413
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005499, upper bound: 0.0005332
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0043262, 0.0033402, 0.0042600, -0.0005056, 0.0004978
1: 0.0018054, 0.0019473, 0.0018049, 0.0019377, -0.0000730, 0.0000719
2: 0.0119680, 0.0125111, 0.0120046, 0.0125132, -0.0002752, 0.0002795
3: -0.0023026, -0.0017409, -0.0022647, -0.0017388, -0.0002847, 0.0002891
4: -0.0021523, -0.0015443, -0.0021547, -0.0015853, -0.0003130, 0.0003082
5: 0.0055748, 0.0061501, 0.0056135, 0.0061524, -0.0002916, 0.0002962
6: -0.0001812, 0.0021016, -0.0000274, 0.0021105, -0.0011571, 0.0011751
7: -0.0054189, -0.0023099, -0.0054310, -0.0025193, -0.0016004, 0.0015759
8: 0.9853967, 0.9875867, 0.9853882, 0.9874392, -0.0011274, 0.0011101
9: -0.0046193, -0.0026313, -0.0044854, -0.0026236, -0.0010077, 0.0010233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005802
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005801
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0043262, 0.0033618, 0.0043421, -0.0004732, 0.0003315
1: 0.0018054, 0.0019473, 0.0018080, 0.0019496, -0.0000684, 0.0000479
2: 0.0119680, 0.0125111, 0.0119592, 0.0125012, -0.0001833, 0.0002616
3: -0.0023026, -0.0017409, -0.0023117, -0.0017511, -0.0001895, 0.0002706
4: -0.0021523, -0.0015443, -0.0021413, -0.0015344, -0.0002929, 0.0002052
5: 0.0055748, 0.0061501, 0.0055654, 0.0061397, -0.0001942, 0.0002772
6: -0.0001812, 0.0021016, -0.0002183, 0.0020602, -0.0007705, 0.0010999
7: -0.0054189, -0.0023099, -0.0053625, -0.0022594, -0.0014980, 0.0010493
8: 0.9853967, 0.9875867, 0.9854364, 0.9876223, -0.0010552, 0.0007391
9: -0.0046193, -0.0026313, -0.0046516, -0.0026674, -0.0006709, 0.0009579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005802
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005800
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033218, 0.0042973, 0.0033412, 0.0042439, -0.0005531, 0.0004930
1: 0.0018022, 0.0019431, 0.0018050, 0.0019354, -0.0000799, 0.0000712
2: 0.0119840, 0.0125233, 0.0120135, 0.0125126, -0.0002726, 0.0003058
3: -0.0022860, -0.0017282, -0.0022555, -0.0017394, -0.0002819, 0.0003163
4: -0.0021660, -0.0015622, -0.0021540, -0.0015952, -0.0003424, 0.0003052
5: 0.0055917, 0.0061632, 0.0056230, 0.0061518, -0.0002888, 0.0003240
6: -0.0001141, 0.0021532, 0.0000100, 0.0021081, -0.0011458, 0.0012855
7: -0.0054892, -0.0024013, -0.0054277, -0.0025704, -0.0017508, 0.0015605
8: 0.9853472, 0.9875223, 0.9853904, 0.9874033, -0.0012333, 0.0010992
9: -0.0045609, -0.0025864, -0.0044528, -0.0026257, -0.0009978, 0.0011195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005771
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005763
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033218, 0.0042973, 0.0033630, 0.0043262, -0.0005397, 0.0003271
1: 0.0018022, 0.0019431, 0.0018082, 0.0019473, -0.0000780, 0.0000473
2: 0.0119840, 0.0125233, 0.0119680, 0.0125006, -0.0001809, 0.0002984
3: -0.0022860, -0.0017282, -0.0023026, -0.0017518, -0.0001871, 0.0003086
4: -0.0021660, -0.0015622, -0.0021405, -0.0015443, -0.0003341, 0.0002025
5: 0.0055917, 0.0061632, 0.0055747, 0.0061390, -0.0001916, 0.0003162
6: -0.0001141, 0.0021532, -0.0001814, 0.0020575, -0.0007603, 0.0012545
7: -0.0054892, -0.0024013, -0.0053588, -0.0023097, -0.0017085, 0.0010355
8: 0.9853472, 0.9875223, 0.9854390, 0.9875869, -0.0012035, 0.0007294
9: -0.0045609, -0.0025864, -0.0046195, -0.0026698, -0.0006621, 0.0010925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005771
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005764
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005413
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005332
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005568, upper bound: 0.0005397
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005387
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005452, upper bound: 0.0005413
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005448, upper bound: 0.0005332
IS_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005520, upper bound: 0.0005397
IS_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005448, upper bound: 0.0005387
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005771
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005764
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005982
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005810, upper bound: 0.0005954
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005778, upper bound: 0.0005771
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005778, upper bound: 0.0005764
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005778, upper bound: 0.0005982
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005778, upper bound: 0.0005955
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005413
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005332
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005413
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005332
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005507, upper bound: 0.0005413
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005499, upper bound: 0.0005332
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005507, upper bound: 0.0005413
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005499, upper bound: 0.0005332
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005802
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005801
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005802
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005800
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005771
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005763
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005771
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.73
Output dim: 8, lower bound: -0.0005930, upper bound: 0.0005764

## BFS IS instance: IS_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033436, 0.0042502, 0.0033778, 0.0042796, -0.0003644, 0.0003480
1: 0.0018054, 0.0019363, 0.0018103, 0.0019406, -0.0000527, 0.0000503
2: 0.0120100, 0.0125113, 0.0119938, 0.0124924, -0.0001924, 0.0002015
3: -0.0022591, -0.0017407, -0.0022759, -0.0017603, -0.0001990, 0.0002084
4: -0.0021525, -0.0015913, -0.0021314, -0.0015731, -0.0002256, 0.0002154
5: 0.0056193, 0.0061504, 0.0056021, 0.0061303, -0.0002039, 0.0002135
6: -0.0000046, 0.0021025, -0.0000729, 0.0020231, -0.0008089, 0.0008471
7: -0.0054201, -0.0025504, -0.0053120, -0.0024574, -0.0011536, 0.0011017
8: 0.9853958, 0.9874173, 0.9854720, 0.9874828, -0.0008126, 0.0007761
9: -0.0044655, -0.0026306, -0.0045250, -0.0026997, -0.0007045, 0.0007377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005611
time: 0.55 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005612
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033226, 0.0042230, 0.0033791, 0.0042651, -0.0004055, 0.0003499
1: 0.0018023, 0.0019324, 0.0018105, 0.0019385, -0.0000586, 0.0000506
2: 0.0120251, 0.0125229, 0.0120018, 0.0124916, -0.0001935, 0.0002242
3: -0.0022436, -0.0017287, -0.0022676, -0.0017610, -0.0002001, 0.0002319
4: -0.0021655, -0.0016082, -0.0021305, -0.0015821, -0.0002510, 0.0002166
5: 0.0056352, 0.0061626, 0.0056106, 0.0061296, -0.0002050, 0.0002376
6: 0.0000585, 0.0021513, -0.0000393, 0.0020200, -0.0008133, 0.0009426
7: -0.0054865, -0.0026364, -0.0053077, -0.0025032, -0.0012837, 0.0011077
8: 0.9853491, 0.9873567, 0.9854749, 0.9874505, -0.0009043, 0.0007803
9: -0.0044106, -0.0025881, -0.0044958, -0.0027024, -0.0007083, 0.0008208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005569
time: 0.57 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005574
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033423, 0.0042582, 0.0034085, 0.0043444, -0.0005348, 0.0004256
1: 0.0018052, 0.0019375, 0.0018147, 0.0019499, -0.0000773, 0.0000615
2: 0.0120056, 0.0125120, 0.0119579, 0.0124754, -0.0002353, 0.0002957
3: -0.0022637, -0.0017399, -0.0023130, -0.0017778, -0.0002434, 0.0003058
4: -0.0021534, -0.0015864, -0.0021124, -0.0015330, -0.0003311, 0.0002635
5: 0.0056146, 0.0061512, 0.0055641, 0.0061124, -0.0002493, 0.0003133
6: -0.0000233, 0.0021057, -0.0002236, 0.0019518, -0.0009893, 0.0012430
7: -0.0054244, -0.0025249, -0.0052149, -0.0022522, -0.0016929, 0.0013473
8: 0.9853928, 0.9874353, 0.9855403, 0.9876274, -0.0011925, 0.0009491
9: -0.0044818, -0.0026278, -0.0046563, -0.0027618, -0.0008615, 0.0010825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005568, upper bound: 0.0005396
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005568, upper bound: 0.0005397
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033436, 0.0042442, 0.0033846, 0.0043203, -0.0005356, 0.0004533
1: 0.0018054, 0.0019355, 0.0018113, 0.0019465, -0.0000774, 0.0000655
2: 0.0120134, 0.0125112, 0.0119713, 0.0124886, -0.0002506, 0.0002961
3: -0.0022557, -0.0017407, -0.0022992, -0.0017641, -0.0002592, 0.0003063
4: -0.0021525, -0.0015951, -0.0021272, -0.0015479, -0.0003316, 0.0002806
5: 0.0056228, 0.0061503, 0.0055782, 0.0061264, -0.0002656, 0.0003138
6: 0.0000094, 0.0021024, -0.0001675, 0.0020073, -0.0010537, 0.0012449
7: -0.0054201, -0.0025696, -0.0052905, -0.0023285, -0.0016955, 0.0014350
8: 0.9853959, 0.9874038, 0.9854872, 0.9875736, -0.0011943, 0.0010109
9: -0.0044533, -0.0026306, -0.0046074, -0.0027135, -0.0009176, 0.0010841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005386
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005387
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033446, 0.0042339, 0.0033568, 0.0042455, -0.0003588, 0.0004285
1: 0.0018055, 0.0019340, 0.0018073, 0.0019357, -0.0000518, 0.0000619
2: 0.0120190, 0.0125107, 0.0120126, 0.0125040, -0.0002369, 0.0001984
3: -0.0022498, -0.0017413, -0.0022564, -0.0017483, -0.0002450, 0.0002051
4: -0.0021519, -0.0016014, -0.0021443, -0.0015943, -0.0002221, 0.0002652
5: 0.0056288, 0.0061498, 0.0056220, 0.0061426, -0.0002510, 0.0002102
6: 0.0000332, 0.0021003, 0.0000063, 0.0020718, -0.0009959, 0.0008339
7: -0.0054171, -0.0026019, -0.0053783, -0.0025653, -0.0011357, 0.0013563
8: 0.9853979, 0.9873810, 0.9854253, 0.9874068, -0.0008000, 0.0009554
9: -0.0044326, -0.0026325, -0.0044560, -0.0026573, -0.0008673, 0.0007262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005460
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005412, upper bound: 0.0005469
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033236, 0.0042069, 0.0033582, 0.0042320, -0.0003995, 0.0004295
1: 0.0018025, 0.0019301, 0.0018075, 0.0019337, -0.0000577, 0.0000621
2: 0.0120340, 0.0125223, 0.0120201, 0.0125032, -0.0002375, 0.0002209
3: -0.0022344, -0.0017293, -0.0022487, -0.0017491, -0.0002456, 0.0002284
4: -0.0021649, -0.0016181, -0.0021435, -0.0016026, -0.0002473, 0.0002659
5: 0.0056446, 0.0061621, 0.0056299, 0.0061418, -0.0002516, 0.0002340
6: 0.0000959, 0.0021490, 0.0000377, 0.0020686, -0.0009983, 0.0009286
7: -0.0054835, -0.0026873, -0.0053740, -0.0026080, -0.0012646, 0.0013596
8: 0.9853512, 0.9873208, 0.9854283, 0.9873767, -0.0008908, 0.0009577
9: -0.0043780, -0.0025900, -0.0044287, -0.0026601, -0.0008693, 0.0008086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005412, upper bound: 0.0005402
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005436
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033432, 0.0042421, 0.0033877, 0.0043118, -0.0005283, 0.0004658
1: 0.0018053, 0.0019352, 0.0018117, 0.0019452, -0.0000763, 0.0000673
2: 0.0120145, 0.0125115, 0.0119760, 0.0124869, -0.0002575, 0.0002921
3: -0.0022545, -0.0017405, -0.0022944, -0.0017659, -0.0002663, 0.0003021
4: -0.0021528, -0.0015963, -0.0021252, -0.0015532, -0.0003270, 0.0002883
5: 0.0056240, 0.0061506, 0.0055832, 0.0061245, -0.0002729, 0.0003095
6: 0.0000142, 0.0021034, -0.0001479, 0.0020000, -0.0010827, 0.0012279
7: -0.0054213, -0.0025760, -0.0052806, -0.0023553, -0.0016723, 0.0014745
8: 0.9853950, 0.9873992, 0.9854941, 0.9875548, -0.0011780, 0.0010387
9: -0.0044492, -0.0026298, -0.0045903, -0.0027198, -0.0009428, 0.0010693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005378, upper bound: 0.0005265
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005379, upper bound: 0.0005281
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033446, 0.0042280, 0.0033620, 0.0042892, -0.0005297, 0.0004926
1: 0.0018055, 0.0019331, 0.0018080, 0.0019420, -0.0000765, 0.0000712
2: 0.0120223, 0.0125107, 0.0119885, 0.0125011, -0.0002723, 0.0002929
3: -0.0022464, -0.0017413, -0.0022814, -0.0017512, -0.0002817, 0.0003029
4: -0.0021519, -0.0016051, -0.0021411, -0.0015672, -0.0003279, 0.0003049
5: 0.0056323, 0.0061498, 0.0055964, 0.0061396, -0.0002885, 0.0003103
6: 0.0000470, 0.0021002, -0.0000954, 0.0020598, -0.0011449, 0.0012312
7: -0.0054170, -0.0026207, -0.0053620, -0.0024268, -0.0016768, 0.0015592
8: 0.9853980, 0.9873678, 0.9854367, 0.9875044, -0.0011812, 0.0010984
9: -0.0044206, -0.0026326, -0.0045446, -0.0026678, -0.0009970, 0.0010722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005256
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033757, 0.0042961, 0.0033420, 0.0042447, -0.0003451, 0.0004345
1: 0.0018100, 0.0019430, 0.0018051, 0.0019355, -0.0000499, 0.0000628
2: 0.0119847, 0.0124935, 0.0120131, 0.0125122, -0.0002402, 0.0001908
3: -0.0022853, -0.0017591, -0.0022559, -0.0017398, -0.0002484, 0.0001973
4: -0.0021326, -0.0015629, -0.0021535, -0.0015948, -0.0002136, 0.0002690
5: 0.0055924, 0.0061315, 0.0056225, 0.0061513, -0.0002545, 0.0002022
6: -0.0001112, 0.0020278, 0.0000083, 0.0021063, -0.0010099, 0.0008022
7: -0.0053185, -0.0024052, -0.0054253, -0.0025680, -0.0010925, 0.0013754
8: 0.9854675, 0.9875196, 0.9853922, 0.9874050, -0.0007696, 0.0009688
9: -0.0045584, -0.0026956, -0.0044543, -0.0026272, -0.0008794, 0.0006986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005565, upper bound: 0.0005590
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005566, upper bound: 0.0005554
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033402, 0.0042600, 0.0033420, 0.0042447, -0.0003188, 0.0003393
1: 0.0018049, 0.0019377, 0.0018051, 0.0019355, -0.0000461, 0.0000490
2: 0.0120046, 0.0125132, 0.0120131, 0.0125122, -0.0001876, 0.0001762
3: -0.0022647, -0.0017388, -0.0022559, -0.0017398, -0.0001940, 0.0001823
4: -0.0021547, -0.0015853, -0.0021535, -0.0015948, -0.0001973, 0.0002100
5: 0.0056135, 0.0061524, 0.0056225, 0.0061513, -0.0001987, 0.0001867
6: -0.0000274, 0.0021105, 0.0000083, 0.0021063, -0.0007886, 0.0007409
7: -0.0054310, -0.0025193, -0.0054253, -0.0025680, -0.0010091, 0.0010740
8: 0.9853882, 0.9874392, 0.9853922, 0.9874050, -0.0007108, 0.0007565
9: -0.0044854, -0.0026236, -0.0044543, -0.0026272, -0.0006867, 0.0006452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005565, upper bound: 0.0005593
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005566, upper bound: 0.0005565
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033757, 0.0042961, 0.0033635, 0.0043261, -0.0004848, 0.0004654
1: 0.0018100, 0.0019430, 0.0018082, 0.0019473, -0.0000700, 0.0000672
2: 0.0119847, 0.0124935, 0.0119681, 0.0125003, -0.0002573, 0.0002680
3: -0.0022853, -0.0017591, -0.0023025, -0.0017521, -0.0002661, 0.0002772
4: -0.0021326, -0.0015629, -0.0021402, -0.0015443, -0.0003001, 0.0002881
5: 0.0055924, 0.0061315, 0.0055748, 0.0061387, -0.0002726, 0.0002840
6: -0.0001112, 0.0020278, -0.0001811, 0.0020563, -0.0010817, 0.0011267
7: -0.0053185, -0.0024052, -0.0053572, -0.0023101, -0.0015345, 0.0014732
8: 0.9854675, 0.9875196, 0.9854401, 0.9875866, -0.0010809, 0.0010378
9: -0.0045584, -0.0026956, -0.0046192, -0.0026708, -0.0009420, 0.0009812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005536, upper bound: 0.0005507
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005499
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033402, 0.0042600, 0.0033635, 0.0043261, -0.0004978, 0.0004053
1: 0.0018049, 0.0019377, 0.0018082, 0.0019473, -0.0000719, 0.0000586
2: 0.0120046, 0.0125132, 0.0119681, 0.0125003, -0.0002241, 0.0002752
3: -0.0022647, -0.0017388, -0.0023025, -0.0017521, -0.0002318, 0.0002847
4: -0.0021547, -0.0015853, -0.0021402, -0.0015443, -0.0003082, 0.0002509
5: 0.0056135, 0.0061524, 0.0055748, 0.0061387, -0.0002375, 0.0002916
6: -0.0000274, 0.0021105, -0.0001811, 0.0020563, -0.0009421, 0.0011571
7: -0.0054310, -0.0025193, -0.0053572, -0.0023101, -0.0015758, 0.0012831
8: 0.9853882, 0.9874392, 0.9854401, 0.9875866, -0.0011101, 0.0009038
9: -0.0044854, -0.0026236, -0.0046192, -0.0026708, -0.0008205, 0.0010076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005536, upper bound: 0.0005675
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005387
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033767, 0.0042771, 0.0033181, 0.0042156, -0.0003400, 0.0005074
1: 0.0018101, 0.0019402, 0.0018017, 0.0019313, -0.0000491, 0.0000733
2: 0.0119951, 0.0124930, 0.0120292, 0.0125254, -0.0002805, 0.0001880
3: -0.0022745, -0.0017597, -0.0022393, -0.0017261, -0.0002901, 0.0001944
4: -0.0021320, -0.0015747, -0.0021683, -0.0016127, -0.0002104, 0.0003141
5: 0.0056035, 0.0061310, 0.0056396, 0.0061653, -0.0002972, 0.0001992
6: -0.0000672, 0.0020255, 0.0000758, 0.0021618, -0.0011794, 0.0007902
7: -0.0053153, -0.0024651, -0.0055010, -0.0026599, -0.0010762, 0.0016062
8: 0.9854696, 0.9874774, 0.9853389, 0.9873402, -0.0007581, 0.0011315
9: -0.0045201, -0.0026976, -0.0043955, -0.0025789, -0.0010271, 0.0006881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005538, upper bound: 0.0005590
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005538, upper bound: 0.0005554
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033412, 0.0042439, 0.0033181, 0.0042156, -0.0003141, 0.0004256
1: 0.0018050, 0.0019354, 0.0018017, 0.0019313, -0.0000454, 0.0000615
2: 0.0120135, 0.0125126, 0.0120292, 0.0125254, -0.0002353, 0.0001737
3: -0.0022555, -0.0017394, -0.0022393, -0.0017261, -0.0002433, 0.0001796
4: -0.0021540, -0.0015952, -0.0021683, -0.0016127, -0.0001944, 0.0002634
5: 0.0056230, 0.0061518, 0.0056396, 0.0061653, -0.0002493, 0.0001840
6: 0.0000100, 0.0021081, 0.0000758, 0.0021618, -0.0009891, 0.0007301
7: -0.0054277, -0.0025704, -0.0055010, -0.0026599, -0.0009943, 0.0013471
8: 0.9853904, 0.9874033, 0.9853389, 0.9873402, -0.0007004, 0.0009489
9: -0.0044528, -0.0026257, -0.0043955, -0.0025789, -0.0008614, 0.0006358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005538, upper bound: 0.0005593
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005538, upper bound: 0.0005565
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033767, 0.0042771, 0.0033403, 0.0042972, -0.0004787, 0.0005247
1: 0.0018101, 0.0019402, 0.0018049, 0.0019431, -0.0000692, 0.0000758
2: 0.0119951, 0.0124930, 0.0119840, 0.0125131, -0.0002901, 0.0002647
3: -0.0022745, -0.0017597, -0.0022860, -0.0017389, -0.0003000, 0.0002737
4: -0.0021320, -0.0015747, -0.0021545, -0.0015622, -0.0002963, 0.0003248
5: 0.0056035, 0.0061310, 0.0055917, 0.0061523, -0.0003074, 0.0002804
6: -0.0000672, 0.0020255, -0.0001140, 0.0021101, -0.0012196, 0.0011127
7: -0.0053153, -0.0024651, -0.0054305, -0.0024015, -0.0015154, 0.0016609
8: 0.9854696, 0.9874774, 0.9853885, 0.9875222, -0.0010675, 0.0011700
9: -0.0045201, -0.0026976, -0.0045608, -0.0026239, -0.0010620, 0.0009690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005469, upper bound: 0.0005397
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005409, upper bound: 0.0005387
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033412, 0.0042439, 0.0033403, 0.0042972, -0.0004930, 0.0004545
1: 0.0018050, 0.0019354, 0.0018049, 0.0019431, -0.0000712, 0.0000657
2: 0.0120135, 0.0125126, 0.0119840, 0.0125131, -0.0002513, 0.0002725
3: -0.0022555, -0.0017394, -0.0022860, -0.0017389, -0.0002599, 0.0002819
4: -0.0021540, -0.0015952, -0.0021545, -0.0015622, -0.0003052, 0.0002813
5: 0.0056230, 0.0061518, 0.0055917, 0.0061523, -0.0002662, 0.0002888
6: 0.0000100, 0.0021081, -0.0001140, 0.0021101, -0.0010563, 0.0011458
7: -0.0054277, -0.0025704, -0.0054305, -0.0024015, -0.0015604, 0.0014386
8: 0.9853904, 0.9874033, 0.9853885, 0.9875222, -0.0010992, 0.0010134
9: -0.0044528, -0.0026257, -0.0045608, -0.0026239, -0.0009199, 0.0009978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005469, upper bound: 0.0005675
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005409, upper bound: 0.0005675
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033677, 0.0043318, 0.0033778, 0.0042796, -0.0004201, 0.0004860
1: 0.0018088, 0.0019481, 0.0018103, 0.0019406, -0.0000607, 0.0000702
2: 0.0119649, 0.0124980, 0.0119938, 0.0124924, -0.0002687, 0.0002322
3: -0.0023058, -0.0017545, -0.0022759, -0.0017603, -0.0002779, 0.0002402
4: -0.0021376, -0.0015408, -0.0021314, -0.0015731, -0.0002600, 0.0003008
5: 0.0055715, 0.0061363, 0.0056021, 0.0061303, -0.0002847, 0.0002461
6: -0.0001943, 0.0020466, -0.0000729, 0.0020231, -0.0011296, 0.0009763
7: -0.0053440, -0.0022922, -0.0053120, -0.0024574, -0.0013297, 0.0015384
8: 0.9854494, 0.9875993, 0.9854720, 0.9874828, -0.0009366, 0.0010837
9: -0.0046307, -0.0026792, -0.0045250, -0.0026997, -0.0009837, 0.0008502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005489
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005489
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0043068, 0.0033791, 0.0042651, -0.0004549, 0.0004869
1: 0.0018054, 0.0019445, 0.0018105, 0.0019385, -0.0000657, 0.0000703
2: 0.0119788, 0.0125111, 0.0120018, 0.0124916, -0.0002692, 0.0002515
3: -0.0022915, -0.0017409, -0.0022676, -0.0017610, -0.0002784, 0.0002601
4: -0.0021523, -0.0015563, -0.0021305, -0.0015821, -0.0002816, 0.0003014
5: 0.0055862, 0.0061501, 0.0056106, 0.0061296, -0.0002852, 0.0002665
6: -0.0001361, 0.0021017, -0.0000393, 0.0020200, -0.0011318, 0.0010573
7: -0.0054190, -0.0023713, -0.0053077, -0.0025032, -0.0014399, 0.0015414
8: 0.9853966, 0.9875435, 0.9854749, 0.9874505, -0.0010143, 0.0010858
9: -0.0045800, -0.0026313, -0.0044958, -0.0027024, -0.0009856, 0.0009207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005425
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005428
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033677, 0.0043318, 0.0034069, 0.0043532, -0.0003733, 0.0003563
1: 0.0018088, 0.0019481, 0.0018145, 0.0019512, -0.0000539, 0.0000515
2: 0.0119649, 0.0124980, 0.0119531, 0.0124762, -0.0001970, 0.0002064
3: -0.0023058, -0.0017545, -0.0023180, -0.0017769, -0.0002037, 0.0002135
4: -0.0021376, -0.0015408, -0.0021133, -0.0015276, -0.0002311, 0.0002205
5: 0.0055715, 0.0061363, 0.0055590, 0.0061133, -0.0002087, 0.0002187
6: -0.0001943, 0.0020466, -0.0002440, 0.0019553, -0.0008280, 0.0008677
7: -0.0053440, -0.0022922, -0.0052197, -0.0022243, -0.0011817, 0.0011277
8: 0.9854494, 0.9875993, 0.9855370, 0.9876471, -0.0008324, 0.0007944
9: -0.0046307, -0.0026792, -0.0046740, -0.0027587, -0.0007211, 0.0007556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005413
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005413
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0043068, 0.0034084, 0.0043403, -0.0004146, 0.0003584
1: 0.0018054, 0.0019445, 0.0018147, 0.0019494, -0.0000599, 0.0000518
2: 0.0119788, 0.0125111, 0.0119602, 0.0124754, -0.0001981, 0.0002292
3: -0.0022915, -0.0017409, -0.0023106, -0.0017778, -0.0002049, 0.0002371
4: -0.0021523, -0.0015563, -0.0021124, -0.0015355, -0.0002566, 0.0002218
5: 0.0055862, 0.0061501, 0.0055665, 0.0061124, -0.0002099, 0.0002428
6: -0.0001361, 0.0021017, -0.0002141, 0.0019519, -0.0008329, 0.0009636
7: -0.0054190, -0.0023713, -0.0052150, -0.0022651, -0.0013123, 0.0011344
8: 0.9853966, 0.9875435, 0.9855403, 0.9876182, -0.0009244, 0.0007991
9: -0.0045800, -0.0026313, -0.0046480, -0.0027617, -0.0007253, 0.0008391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005329
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005332
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033688, 0.0043156, 0.0033568, 0.0042455, -0.0004143, 0.0005588
1: 0.0018090, 0.0019458, 0.0018073, 0.0019357, -0.0000599, 0.0000807
2: 0.0119739, 0.0124973, 0.0120126, 0.0125040, -0.0003090, 0.0002290
3: -0.0022965, -0.0017551, -0.0022564, -0.0017483, -0.0003196, 0.0002369
4: -0.0021369, -0.0015509, -0.0021443, -0.0015943, -0.0002564, 0.0003459
5: 0.0055810, 0.0061356, 0.0056220, 0.0061426, -0.0003274, 0.0002427
6: -0.0001566, 0.0020440, 0.0000063, 0.0020718, -0.0012989, 0.0009629
7: -0.0053404, -0.0023435, -0.0053783, -0.0025653, -0.0013114, 0.0017690
8: 0.9854520, 0.9875631, 0.9854253, 0.9874068, -0.0009238, 0.0012461
9: -0.0045979, -0.0026816, -0.0044560, -0.0026573, -0.0011312, 0.0008385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005380, upper bound: 0.0005355
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005381, upper bound: 0.0005359
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033450, 0.0042918, 0.0033582, 0.0042320, -0.0004488, 0.0005591
1: 0.0018056, 0.0019423, 0.0018075, 0.0019337, -0.0000648, 0.0000808
2: 0.0119871, 0.0125105, 0.0120201, 0.0125032, -0.0003091, 0.0002481
3: -0.0022829, -0.0017415, -0.0022487, -0.0017491, -0.0003197, 0.0002566
4: -0.0021517, -0.0015656, -0.0021435, -0.0016026, -0.0002778, 0.0003461
5: 0.0055949, 0.0061495, 0.0056299, 0.0061418, -0.0003275, 0.0002629
6: -0.0001012, 0.0020993, 0.0000377, 0.0020686, -0.0012994, 0.0010432
7: -0.0054157, -0.0024188, -0.0053740, -0.0026080, -0.0014207, 0.0017697
8: 0.9853988, 0.9875100, 0.9854283, 0.9873767, -0.0010008, 0.0012466
9: -0.0045497, -0.0026334, -0.0044287, -0.0026601, -0.0011316, 0.0009085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005371, upper bound: 0.0005287
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005371, upper bound: 0.0005307
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033688, 0.0043156, 0.0033862, 0.0043201, -0.0003685, 0.0004370
1: 0.0018090, 0.0019458, 0.0018115, 0.0019464, -0.0000532, 0.0000631
2: 0.0119739, 0.0124973, 0.0119714, 0.0124877, -0.0002416, 0.0002037
3: -0.0022965, -0.0017551, -0.0022991, -0.0017651, -0.0002499, 0.0002107
4: -0.0021369, -0.0015509, -0.0021261, -0.0015480, -0.0002281, 0.0002705
5: 0.0055810, 0.0061356, 0.0055783, 0.0061254, -0.0002560, 0.0002159
6: -0.0001566, 0.0020440, -0.0001672, 0.0020035, -0.0010157, 0.0008565
7: -0.0053404, -0.0023435, -0.0052852, -0.0023290, -0.0011665, 0.0013833
8: 0.9854520, 0.9875631, 0.9854908, 0.9875732, -0.0008217, 0.0009744
9: -0.0045979, -0.0026816, -0.0046071, -0.0027168, -0.0008845, 0.0007459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005380, upper bound: 0.0005289
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005381, upper bound: 0.0005285
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033450, 0.0042918, 0.0033877, 0.0043080, -0.0004092, 0.0004384
1: 0.0018056, 0.0019423, 0.0018117, 0.0019447, -0.0000591, 0.0000633
2: 0.0119871, 0.0125105, 0.0119781, 0.0124869, -0.0002424, 0.0002262
3: -0.0022829, -0.0017415, -0.0022922, -0.0017659, -0.0002507, 0.0002340
4: -0.0021517, -0.0015656, -0.0021253, -0.0015555, -0.0002533, 0.0002714
5: 0.0055949, 0.0061495, 0.0055854, 0.0061246, -0.0002568, 0.0002397
6: -0.0001012, 0.0020993, -0.0001391, 0.0020001, -0.0010190, 0.0009510
7: -0.0054157, -0.0024188, -0.0052807, -0.0023673, -0.0012952, 0.0013878
8: 0.9853988, 0.9875100, 0.9854940, 0.9875464, -0.0009124, 0.0009776
9: -0.0045497, -0.0026334, -0.0045826, -0.0027197, -0.0008874, 0.0008282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005371, upper bound: 0.0005204
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005371, upper bound: 0.0005210
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0034066, 0.0043551, 0.0033402, 0.0042600, -0.0004319, 0.0005929
1: 0.0018145, 0.0019515, 0.0018049, 0.0019377, -0.0000624, 0.0000857
2: 0.0119521, 0.0124764, 0.0120046, 0.0125132, -0.0003278, 0.0002388
3: -0.0023191, -0.0017768, -0.0022647, -0.0017388, -0.0003390, 0.0002469
4: -0.0021135, -0.0015264, -0.0021547, -0.0015853, -0.0002673, 0.0003670
5: 0.0055579, 0.0061135, 0.0056135, 0.0061524, -0.0003473, 0.0002530
6: -0.0002484, 0.0019561, -0.0000274, 0.0021105, -0.0013782, 0.0010038
7: -0.0052207, -0.0022185, -0.0054310, -0.0025193, -0.0013670, 0.0018769
8: 0.9855363, 0.9876511, 0.9853882, 0.9874392, -0.0009630, 0.0013222
9: -0.0046778, -0.0027581, -0.0044854, -0.0026236, -0.0012002, 0.0008741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005568
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005485
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033635, 0.0043261, 0.0033402, 0.0042600, -0.0004053, 0.0004978
1: 0.0018082, 0.0019473, 0.0018049, 0.0019377, -0.0000586, 0.0000719
2: 0.0119681, 0.0125003, 0.0120046, 0.0125132, -0.0002752, 0.0002241
3: -0.0023025, -0.0017521, -0.0022647, -0.0017388, -0.0002847, 0.0002318
4: -0.0021402, -0.0015443, -0.0021547, -0.0015853, -0.0002509, 0.0003082
5: 0.0055748, 0.0061387, 0.0056135, 0.0061524, -0.0002916, 0.0002375
6: -0.0001811, 0.0020563, -0.0000274, 0.0021105, -0.0011571, 0.0009421
7: -0.0053572, -0.0023101, -0.0054310, -0.0025193, -0.0012831, 0.0015758
8: 0.9854401, 0.9875866, 0.9853882, 0.9874392, -0.0009038, 0.0011101
9: -0.0046192, -0.0026708, -0.0044854, -0.0026236, -0.0010076, 0.0008205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005624
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005575
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0034066, 0.0043551, 0.0033618, 0.0043421, -0.0003733, 0.0004216
1: 0.0018145, 0.0019515, 0.0018080, 0.0019496, -0.0000539, 0.0000609
2: 0.0119521, 0.0124764, 0.0119592, 0.0125012, -0.0002331, 0.0002064
3: -0.0023191, -0.0017768, -0.0023117, -0.0017511, -0.0002411, 0.0002135
4: -0.0021135, -0.0015264, -0.0021413, -0.0015344, -0.0002311, 0.0002610
5: 0.0055579, 0.0061135, 0.0055654, 0.0061397, -0.0002470, 0.0002187
6: -0.0002484, 0.0019561, -0.0002183, 0.0020602, -0.0009800, 0.0008677
7: -0.0052207, -0.0022185, -0.0053625, -0.0022594, -0.0011818, 0.0013346
8: 0.9855363, 0.9876511, 0.9854364, 0.9876223, -0.0008325, 0.0009402
9: -0.0046778, -0.0027581, -0.0046516, -0.0026674, -0.0008534, 0.0007557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005437
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005437
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033635, 0.0043261, 0.0033618, 0.0043421, -0.0003518, 0.0003315
1: 0.0018082, 0.0019473, 0.0018080, 0.0019496, -0.0000508, 0.0000479
2: 0.0119681, 0.0125003, 0.0119592, 0.0125012, -0.0001833, 0.0001945
3: -0.0023025, -0.0017521, -0.0023117, -0.0017511, -0.0001895, 0.0002012
4: -0.0021402, -0.0015443, -0.0021413, -0.0015344, -0.0002178, 0.0002052
5: 0.0055748, 0.0061387, 0.0055654, 0.0061397, -0.0001942, 0.0002061
6: -0.0001811, 0.0020563, -0.0002183, 0.0020602, -0.0007704, 0.0008178
7: -0.0053572, -0.0023101, -0.0053625, -0.0022594, -0.0011137, 0.0010492
8: 0.9854401, 0.9875866, 0.9854364, 0.9876223, -0.0007845, 0.0007391
9: -0.0046192, -0.0026708, -0.0046516, -0.0026674, -0.0006709, 0.0007121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005551
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005551
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033859, 0.0043220, 0.0033412, 0.0042439, -0.0004716, 0.0005869
1: 0.0018115, 0.0019467, 0.0018050, 0.0019354, -0.0000681, 0.0000848
2: 0.0119703, 0.0124879, 0.0120135, 0.0125126, -0.0003245, 0.0002607
3: -0.0023002, -0.0017649, -0.0022555, -0.0017394, -0.0003356, 0.0002697
4: -0.0021263, -0.0015469, -0.0021540, -0.0015952, -0.0002919, 0.0003633
5: 0.0055772, 0.0061256, 0.0056230, 0.0061518, -0.0003438, 0.0002763
6: -0.0001716, 0.0020042, 0.0000100, 0.0021081, -0.0013642, 0.0010961
7: -0.0052862, -0.0023230, -0.0054277, -0.0025704, -0.0014928, 0.0018579
8: 0.9854901, 0.9875774, 0.9853904, 0.9874033, -0.0010516, 0.0013088
9: -0.0046109, -0.0027162, -0.0044528, -0.0026257, -0.0011880, 0.0009545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005519
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005449
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033403, 0.0042972, 0.0033412, 0.0042439, -0.0004545, 0.0004930
1: 0.0018049, 0.0019431, 0.0018050, 0.0019354, -0.0000657, 0.0000712
2: 0.0119840, 0.0125131, 0.0120135, 0.0125126, -0.0002725, 0.0002513
3: -0.0022860, -0.0017389, -0.0022555, -0.0017394, -0.0002819, 0.0002599
4: -0.0021545, -0.0015622, -0.0021540, -0.0015952, -0.0002813, 0.0003052
5: 0.0055917, 0.0061523, 0.0056230, 0.0061518, -0.0002888, 0.0002662
6: -0.0001140, 0.0021101, 0.0000100, 0.0021081, -0.0011458, 0.0010563
7: -0.0054305, -0.0024015, -0.0054277, -0.0025704, -0.0014386, 0.0015604
8: 0.9853885, 0.9875222, 0.9853904, 0.9874033, -0.0010134, 0.0010992
9: -0.0045608, -0.0026239, -0.0044528, -0.0026257, -0.0009978, 0.0009199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005576
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005546
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033859, 0.0043220, 0.0033630, 0.0043262, -0.0004518, 0.0004168
1: 0.0018115, 0.0019467, 0.0018082, 0.0019473, -0.0000653, 0.0000602
2: 0.0119703, 0.0124879, 0.0119680, 0.0125006, -0.0002304, 0.0002498
3: -0.0023002, -0.0017649, -0.0023026, -0.0017518, -0.0002383, 0.0002584
4: -0.0021263, -0.0015469, -0.0021405, -0.0015443, -0.0002797, 0.0002580
5: 0.0055772, 0.0061256, 0.0055747, 0.0061390, -0.0002442, 0.0002647
6: -0.0001716, 0.0020042, -0.0001814, 0.0020575, -0.0009688, 0.0010501
7: -0.0052862, -0.0023230, -0.0053588, -0.0023097, -0.0014302, 0.0013194
8: 0.9854901, 0.9875774, 0.9854390, 0.9875869, -0.0010075, 0.0009294
9: -0.0046109, -0.0027162, -0.0046195, -0.0026698, -0.0008436, 0.0009145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005406
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005404
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033403, 0.0042972, 0.0033630, 0.0043262, -0.0004374, 0.0003271
1: 0.0018049, 0.0019431, 0.0018082, 0.0019473, -0.0000632, 0.0000473
2: 0.0119840, 0.0125131, 0.0119680, 0.0125006, -0.0001808, 0.0002418
3: -0.0022860, -0.0017389, -0.0023026, -0.0017518, -0.0001870, 0.0002501
4: -0.0021545, -0.0015622, -0.0021405, -0.0015443, -0.0002708, 0.0002025
5: 0.0055917, 0.0061523, 0.0055747, 0.0061390, -0.0001916, 0.0002562
6: -0.0001140, 0.0021101, -0.0001814, 0.0020575, -0.0007603, 0.0010166
7: -0.0054305, -0.0024015, -0.0053588, -0.0023097, -0.0013846, 0.0010354
8: 0.9853885, 0.9875222, 0.9854390, 0.9875869, -0.0009753, 0.0007294
9: -0.0045608, -0.0026239, -0.0046195, -0.0026698, -0.0006621, 0.0008853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005508
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005508
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.98 seconds
IS_A1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005611
IS_A1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005612
IS_A1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005569
IS_A1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005585, upper bound: 0.0005574
IS_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005568, upper bound: 0.0005396
IS_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005568, upper bound: 0.0005397
IS_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005386
IS_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005485, upper bound: 0.0005387
IS_A1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005460
IS_A1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005412, upper bound: 0.0005469
IS_A1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005412, upper bound: 0.0005402
IS_A1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005436
IS_A1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005378, upper bound: 0.0005265
IS_A1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005379, upper bound: 0.0005281
IS_A1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005256
IS_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
IS_A1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005565, upper bound: 0.0005590
IS_A1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005566, upper bound: 0.0005554
IS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005565, upper bound: 0.0005593
IS_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005566, upper bound: 0.0005565
IS_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005536, upper bound: 0.0005507
IS_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005499
IS_A1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005536, upper bound: 0.0005675
IS_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005387
IS_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005538, upper bound: 0.0005590
IS_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005538, upper bound: 0.0005554
IS_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005538, upper bound: 0.0005593
IS_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005538, upper bound: 0.0005565
IS_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005469, upper bound: 0.0005397
IS_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005409, upper bound: 0.0005387
IS_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005469, upper bound: 0.0005675
IS_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005409, upper bound: 0.0005675
IS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005489
IS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005489
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005425
IS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005428
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005413
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005549, upper bound: 0.0005413
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005329
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005545, upper bound: 0.0005332
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005380, upper bound: 0.0005355
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005381, upper bound: 0.0005359
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005371, upper bound: 0.0005287
IS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005371, upper bound: 0.0005307
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005380, upper bound: 0.0005289
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005381, upper bound: 0.0005285
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005371, upper bound: 0.0005204
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005371, upper bound: 0.0005210
IS_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005568
IS_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005485
IS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005624
IS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005575
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005437
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005437
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005551
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005551
IS_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005519
IS_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005449
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005576
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005546
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005406
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005404
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005508
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 8, lower bound: -0.0005387, upper bound: 0.0005508

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033453, 0.0042347, 0.0033778, 0.0042796, -0.0003629, 0.0003267
1: 0.0018056, 0.0019341, 0.0018103, 0.0019406, -0.0000524, 0.0000472
2: 0.0120186, 0.0125103, 0.0119938, 0.0124924, -0.0001806, 0.0002006
3: -0.0022502, -0.0017417, -0.0022759, -0.0017603, -0.0001868, 0.0002075
4: -0.0021515, -0.0016010, -0.0021314, -0.0015731, -0.0002246, 0.0002022
5: 0.0056284, 0.0061494, 0.0056021, 0.0061303, -0.0001914, 0.0002126
6: 0.0000315, 0.0020985, -0.0000729, 0.0020231, -0.0007593, 0.0008435
7: -0.0054147, -0.0025996, -0.0053120, -0.0024574, -0.0011487, 0.0010341
8: 0.9853997, 0.9873827, 0.9854720, 0.9874828, -0.0008092, 0.0007284
9: -0.0044341, -0.0026340, -0.0045250, -0.0026997, -0.0006612, 0.0007345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005460
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005469
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033224, 0.0042053, 0.0033778, 0.0042796, -0.0004368, 0.0003385
1: 0.0018023, 0.0019299, 0.0018103, 0.0019406, -0.0000631, 0.0000489
2: 0.0120348, 0.0125230, 0.0119938, 0.0124924, -0.0001872, 0.0002415
3: -0.0022335, -0.0017286, -0.0022759, -0.0017603, -0.0001936, 0.0002498
4: -0.0021657, -0.0016191, -0.0021314, -0.0015731, -0.0002704, 0.0002096
5: 0.0056456, 0.0061628, 0.0056021, 0.0061303, -0.0001983, 0.0002559
6: 0.0000996, 0.0021519, -0.0000729, 0.0020231, -0.0007869, 0.0010153
7: -0.0054874, -0.0026924, -0.0053120, -0.0024574, -0.0013828, 0.0010716
8: 0.9853485, 0.9873173, 0.9854720, 0.9874828, -0.0009740, 0.0007549
9: -0.0043748, -0.0025876, -0.0045250, -0.0026997, -0.0006852, 0.0008842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005469
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005469
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033242, 0.0042076, 0.0033791, 0.0042651, -0.0004039, 0.0003291
1: 0.0018026, 0.0019302, 0.0018105, 0.0019385, -0.0000584, 0.0000475
2: 0.0120336, 0.0125220, 0.0120018, 0.0124916, -0.0001819, 0.0002233
3: -0.0022348, -0.0017296, -0.0022676, -0.0017610, -0.0001882, 0.0002310
4: -0.0021645, -0.0016177, -0.0021305, -0.0015821, -0.0002500, 0.0002037
5: 0.0056442, 0.0061617, 0.0056106, 0.0061296, -0.0001928, 0.0002366
6: 0.0000944, 0.0021476, -0.0000393, 0.0020200, -0.0007649, 0.0009389
7: -0.0054816, -0.0026852, -0.0053077, -0.0025032, -0.0012787, 0.0010417
8: 0.9853525, 0.9873224, 0.9854749, 0.9874505, -0.0009007, 0.0007338
9: -0.0043793, -0.0025913, -0.0044958, -0.0027024, -0.0006661, 0.0008176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005402
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005433
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0032995, 0.0041798, 0.0033791, 0.0042651, -0.0004647, 0.0003400
1: 0.0017990, 0.0019262, 0.0018105, 0.0019385, -0.0000671, 0.0000491
2: 0.0120490, 0.0125357, 0.0120018, 0.0124916, -0.0001880, 0.0002569
3: -0.0022189, -0.0017155, -0.0022676, -0.0017610, -0.0001944, 0.0002657
4: -0.0021799, -0.0016349, -0.0021305, -0.0015821, -0.0002876, 0.0002105
5: 0.0056605, 0.0061762, 0.0056106, 0.0061296, -0.0001992, 0.0002722
6: 0.0001590, 0.0022051, -0.0000393, 0.0020200, -0.0007902, 0.0010800
7: -0.0055599, -0.0027733, -0.0053077, -0.0025032, -0.0014708, 0.0010762
8: 0.9852974, 0.9872603, 0.9854749, 0.9874505, -0.0010361, 0.0007581
9: -0.0043231, -0.0025412, -0.0044958, -0.0027024, -0.0006882, 0.0009405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005402
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005436
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0042429, 0.0034085, 0.0043444, -0.0005332, 0.0004043
1: 0.0018054, 0.0019353, 0.0018147, 0.0019499, -0.0000770, 0.0000584
2: 0.0120141, 0.0125110, 0.0119579, 0.0124754, -0.0002235, 0.0002948
3: -0.0022549, -0.0017410, -0.0023130, -0.0017778, -0.0002312, 0.0003049
4: -0.0021523, -0.0015959, -0.0021124, -0.0015330, -0.0003301, 0.0002503
5: 0.0056236, 0.0061501, 0.0055641, 0.0061124, -0.0002369, 0.0003124
6: 0.0000124, 0.0021016, -0.0002236, 0.0019518, -0.0009398, 0.0012394
7: -0.0054189, -0.0025736, -0.0052149, -0.0022522, -0.0016880, 0.0012799
8: 0.9853967, 0.9874010, 0.9855403, 0.9876274, -0.0011890, 0.0009016
9: -0.0044507, -0.0026314, -0.0046563, -0.0027618, -0.0008184, 0.0010793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005265
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005424, upper bound: 0.0005279
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033211, 0.0042137, 0.0034085, 0.0043444, -0.0006071, 0.0004142
1: 0.0018021, 0.0019311, 0.0018147, 0.0019499, -0.0000877, 0.0000598
2: 0.0120302, 0.0125237, 0.0119579, 0.0124754, -0.0002290, 0.0003357
3: -0.0022383, -0.0017278, -0.0023130, -0.0017778, -0.0002369, 0.0003472
4: -0.0021665, -0.0016139, -0.0021124, -0.0015330, -0.0003758, 0.0002564
5: 0.0056406, 0.0061636, 0.0055641, 0.0061124, -0.0002427, 0.0003557
6: 0.0000801, 0.0021548, -0.0002236, 0.0019518, -0.0009628, 0.0014111
7: -0.0054914, -0.0026658, -0.0052149, -0.0022522, -0.0019219, 0.0013113
8: 0.9853456, 0.9873360, 0.9855403, 0.9876274, -0.0013538, 0.0009237
9: -0.0043918, -0.0025850, -0.0046563, -0.0027618, -0.0008385, 0.0012289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005445, upper bound: 0.0005281
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005424, upper bound: 0.0005281
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033454, 0.0042288, 0.0033846, 0.0043203, -0.0005341, 0.0004318
1: 0.0018056, 0.0019332, 0.0018113, 0.0019465, -0.0000772, 0.0000624
2: 0.0120219, 0.0125103, 0.0119713, 0.0124886, -0.0002387, 0.0002953
3: -0.0022469, -0.0017417, -0.0022992, -0.0017641, -0.0002469, 0.0003054
4: -0.0021514, -0.0016046, -0.0021272, -0.0015479, -0.0003306, 0.0002673
5: 0.0056318, 0.0061493, 0.0055782, 0.0061264, -0.0002529, 0.0003129
6: 0.0000451, 0.0020984, -0.0001675, 0.0020073, -0.0010035, 0.0012413
7: -0.0054145, -0.0026181, -0.0052905, -0.0023285, -0.0016905, 0.0013667
8: 0.9853998, 0.9873697, 0.9854872, 0.9875736, -0.0011909, 0.0009627
9: -0.0044223, -0.0026342, -0.0046074, -0.0027135, -0.0008739, 0.0010810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005364, upper bound: 0.0005267
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005357, upper bound: 0.0005267
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033225, 0.0042001, 0.0033846, 0.0043203, -0.0006079, 0.0004374
1: 0.0018023, 0.0019291, 0.0018113, 0.0019465, -0.0000878, 0.0000632
2: 0.0120377, 0.0125230, 0.0119713, 0.0124886, -0.0002418, 0.0003361
3: -0.0022305, -0.0017286, -0.0022992, -0.0017641, -0.0002501, 0.0003476
4: -0.0021656, -0.0016223, -0.0021272, -0.0015479, -0.0003763, 0.0002707
5: 0.0056486, 0.0061627, 0.0055782, 0.0061264, -0.0002562, 0.0003561
6: 0.0001117, 0.0021516, -0.0001675, 0.0020073, -0.0010166, 0.0014128
7: -0.0054871, -0.0027089, -0.0052905, -0.0023285, -0.0019242, 0.0013845
8: 0.9853487, 0.9873056, 0.9854872, 0.9875736, -0.0013554, 0.0009753
9: -0.0043642, -0.0025878, -0.0046074, -0.0027135, -0.0008853, 0.0012304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005364, upper bound: 0.0005269
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005357, upper bound: 0.0005269
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033456, 0.0042215, 0.0033572, 0.0042401, -0.0003461, 0.0004131
1: 0.0018056, 0.0019322, 0.0018073, 0.0019349, -0.0000500, 0.0000597
2: 0.0120259, 0.0125102, 0.0120156, 0.0125038, -0.0002284, 0.0001914
3: -0.0022427, -0.0017418, -0.0022533, -0.0017485, -0.0002362, 0.0001979
4: -0.0021513, -0.0016091, -0.0021441, -0.0015976, -0.0002143, 0.0002557
5: 0.0056361, 0.0061492, 0.0056252, 0.0061424, -0.0002420, 0.0002028
6: 0.0000621, 0.0020980, 0.0000190, 0.0020710, -0.0009602, 0.0008045
7: -0.0054140, -0.0026413, -0.0053772, -0.0025825, -0.0010957, 0.0013077
8: 0.9854001, 0.9873533, 0.9854261, 0.9873947, -0.0007718, 0.0009211
9: -0.0044074, -0.0026345, -0.0044450, -0.0026580, -0.0008362, 0.0007006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005460
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005460
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033436, 0.0042191, 0.0033571, 0.0042396, -0.0003597, 0.0004173
1: 0.0018054, 0.0019318, 0.0018073, 0.0019348, -0.0000520, 0.0000603
2: 0.0120272, 0.0125113, 0.0120159, 0.0125038, -0.0002307, 0.0001988
3: -0.0022414, -0.0017407, -0.0022530, -0.0017484, -0.0002386, 0.0002057
4: -0.0021525, -0.0016106, -0.0021442, -0.0015979, -0.0002226, 0.0002583
5: 0.0056375, 0.0061504, 0.0056255, 0.0061425, -0.0002445, 0.0002107
6: 0.0000676, 0.0021026, 0.0000201, 0.0020711, -0.0009700, 0.0008359
7: -0.0054202, -0.0026487, -0.0053774, -0.0025841, -0.0011385, 0.0013211
8: 0.9853958, 0.9873480, 0.9854259, 0.9873935, -0.0008020, 0.0009306
9: -0.0044027, -0.0026305, -0.0044440, -0.0026579, -0.0008447, 0.0007280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005469
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005469
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033248, 0.0041943, 0.0033586, 0.0042264, -0.0003875, 0.0004144
1: 0.0018026, 0.0019283, 0.0018075, 0.0019329, -0.0000560, 0.0000599
2: 0.0120409, 0.0125217, 0.0120232, 0.0125030, -0.0002291, 0.0002142
3: -0.0022272, -0.0017299, -0.0022455, -0.0017493, -0.0002369, 0.0002216
4: -0.0021642, -0.0016259, -0.0021433, -0.0016061, -0.0002399, 0.0002565
5: 0.0056520, 0.0061614, 0.0056332, 0.0061416, -0.0002427, 0.0002270
6: 0.0001253, 0.0021463, 0.0000507, 0.0020678, -0.0009631, 0.0009006
7: -0.0054798, -0.0027273, -0.0053729, -0.0026258, -0.0012266, 0.0013116
8: 0.9853537, 0.9872927, 0.9854292, 0.9873642, -0.0008640, 0.0009240
9: -0.0043524, -0.0025924, -0.0044173, -0.0026608, -0.0008387, 0.0007843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005402
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005402
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033213, 0.0041926, 0.0033585, 0.0042264, -0.0003991, 0.0004198
1: 0.0018021, 0.0019280, 0.0018075, 0.0019329, -0.0000577, 0.0000606
2: 0.0120419, 0.0125236, 0.0120232, 0.0125030, -0.0002321, 0.0002206
3: -0.0022262, -0.0017280, -0.0022455, -0.0017492, -0.0002400, 0.0002282
4: -0.0021663, -0.0016270, -0.0021433, -0.0016061, -0.0002470, 0.0002598
5: 0.0056530, 0.0061634, 0.0056332, 0.0061417, -0.0002459, 0.0002338
6: 0.0001292, 0.0021543, 0.0000507, 0.0020680, -0.0009756, 0.0009276
7: -0.0054907, -0.0027327, -0.0053731, -0.0026258, -0.0012633, 0.0013287
8: 0.9853461, 0.9872888, 0.9854289, 0.9873642, -0.0008899, 0.0009360
9: -0.0043490, -0.0025855, -0.0044174, -0.0026606, -0.0008496, 0.0008078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005436
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005436
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033442, 0.0042296, 0.0033881, 0.0043066, -0.0005190, 0.0004500
1: 0.0018054, 0.0019333, 0.0018118, 0.0019445, -0.0000750, 0.0000650
2: 0.0120215, 0.0125109, 0.0119789, 0.0124867, -0.0002488, 0.0002870
3: -0.0022473, -0.0017410, -0.0022914, -0.0017662, -0.0002573, 0.0002968
4: -0.0021522, -0.0016041, -0.0021250, -0.0015564, -0.0003213, 0.0002786
5: 0.0056314, 0.0061500, 0.0055863, 0.0061243, -0.0002636, 0.0003040
6: 0.0000433, 0.0021012, -0.0001357, 0.0019991, -0.0010459, 0.0012064
7: -0.0054183, -0.0026157, -0.0052793, -0.0023719, -0.0016430, 0.0014245
8: 0.9853971, 0.9873713, 0.9854950, 0.9875431, -0.0011573, 0.0010034
9: -0.0044238, -0.0026317, -0.0045797, -0.0027206, -0.0009108, 0.0010505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005338, upper bound: 0.0005265
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005338, upper bound: 0.0005265
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033422, 0.0042273, 0.0033880, 0.0043058, -0.0005239, 0.0004541
1: 0.0018051, 0.0019330, 0.0018118, 0.0019444, -0.0000757, 0.0000656
2: 0.0120227, 0.0125121, 0.0119793, 0.0124867, -0.0002510, 0.0002897
3: -0.0022460, -0.0017399, -0.0022909, -0.0017661, -0.0002596, 0.0002996
4: -0.0021534, -0.0016055, -0.0021250, -0.0015569, -0.0003243, 0.0002811
5: 0.0056327, 0.0061512, 0.0055867, 0.0061243, -0.0002660, 0.0003069
6: 0.0000486, 0.0021058, -0.0001339, 0.0019993, -0.0010554, 0.0012178
7: -0.0054247, -0.0026228, -0.0052796, -0.0023743, -0.0016585, 0.0014373
8: 0.9853926, 0.9873663, 0.9854948, 0.9875413, -0.0011683, 0.0010125
9: -0.0044192, -0.0026276, -0.0045781, -0.0027204, -0.0009191, 0.0010605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005338, upper bound: 0.0005281
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005338, upper bound: 0.0005281
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033456, 0.0042155, 0.0033625, 0.0042838, -0.0005205, 0.0004759
1: 0.0018056, 0.0019313, 0.0018081, 0.0019412, -0.0000752, 0.0000687
2: 0.0120292, 0.0125102, 0.0119914, 0.0125008, -0.0002631, 0.0002878
3: -0.0022393, -0.0017419, -0.0022783, -0.0017515, -0.0002721, 0.0002976
4: -0.0021513, -0.0016128, -0.0021408, -0.0015705, -0.0003222, 0.0002946
5: 0.0056396, 0.0061492, 0.0055996, 0.0061393, -0.0002788, 0.0003049
6: 0.0000759, 0.0020979, -0.0000828, 0.0020587, -0.0011060, 0.0012097
7: -0.0054139, -0.0026601, -0.0053604, -0.0024440, -0.0016476, 0.0015063
8: 0.9854002, 0.9873400, 0.9854378, 0.9874923, -0.0011606, 0.0010611
9: -0.0043954, -0.0026346, -0.0045336, -0.0026687, -0.0009632, 0.0010535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005285, upper bound: 0.0005256
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005285, upper bound: 0.0005256
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033436, 0.0042133, 0.0033623, 0.0042835, -0.0005250, 0.0004806
1: 0.0018054, 0.0019310, 0.0018081, 0.0019411, -0.0000759, 0.0000694
2: 0.0120305, 0.0125113, 0.0119916, 0.0125009, -0.0002657, 0.0002903
3: -0.0022380, -0.0017407, -0.0022781, -0.0017514, -0.0002748, 0.0003002
4: -0.0021525, -0.0016142, -0.0021410, -0.0015707, -0.0003250, 0.0002975
5: 0.0056409, 0.0061504, 0.0055998, 0.0061394, -0.0002815, 0.0003076
6: 0.0000812, 0.0021026, -0.0000820, 0.0020591, -0.0011169, 0.0012203
7: -0.0054202, -0.0026673, -0.0053610, -0.0024451, -0.0016620, 0.0015212
8: 0.9853957, 0.9873350, 0.9854374, 0.9874915, -0.0011707, 0.0010715
9: -0.0043908, -0.0026305, -0.0045329, -0.0026684, -0.0009727, 0.0010627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005285, upper bound: 0.0005269
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005285, upper bound: 0.0005269
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033774, 0.0042865, 0.0033423, 0.0042429, -0.0003385, 0.0004175
1: 0.0018102, 0.0019416, 0.0018052, 0.0019353, -0.0000489, 0.0000603
2: 0.0119900, 0.0124926, 0.0120141, 0.0125120, -0.0002308, 0.0001871
3: -0.0022799, -0.0017600, -0.0022549, -0.0017399, -0.0002388, 0.0001936
4: -0.0021316, -0.0015689, -0.0021534, -0.0015958, -0.0002095, 0.0002585
5: 0.0055980, 0.0061306, 0.0056236, 0.0061512, -0.0002446, 0.0001983
6: -0.0000890, 0.0020240, 0.0000123, 0.0021057, -0.0009705, 0.0007867
7: -0.0053132, -0.0024354, -0.0054245, -0.0025735, -0.0010715, 0.0013217
8: 0.9854712, 0.9874983, 0.9853927, 0.9874010, -0.0007548, 0.0009310
9: -0.0045391, -0.0026989, -0.0044508, -0.0026278, -0.0008451, 0.0006851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005620, upper bound: 0.0005590
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005620, upper bound: 0.0005590
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033564, 0.0042589, 0.0033437, 0.0042289, -0.0003790, 0.0004209
1: 0.0018072, 0.0019376, 0.0018054, 0.0019332, -0.0000547, 0.0000608
2: 0.0120052, 0.0125042, 0.0120218, 0.0125112, -0.0002327, 0.0002095
3: -0.0022641, -0.0017480, -0.0022469, -0.0017408, -0.0002406, 0.0002167
4: -0.0021446, -0.0015859, -0.0021525, -0.0016046, -0.0002346, 0.0002605
5: 0.0056142, 0.0061429, 0.0056318, 0.0061503, -0.0002465, 0.0002220
6: -0.0000249, 0.0020729, 0.0000450, 0.0021024, -0.0009782, 0.0008808
7: -0.0053798, -0.0025228, -0.0054199, -0.0026180, -0.0011996, 0.0013322
8: 0.9854242, 0.9874368, 0.9853960, 0.9873697, -0.0008450, 0.0009384
9: -0.0044832, -0.0026564, -0.0044223, -0.0026307, -0.0008518, 0.0007671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005622, upper bound: 0.0005546
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005622, upper bound: 0.0005554
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033418, 0.0042502, 0.0033423, 0.0042429, -0.0003119, 0.0003223
1: 0.0018051, 0.0019363, 0.0018052, 0.0019353, -0.0000451, 0.0000466
2: 0.0120100, 0.0125122, 0.0120141, 0.0125120, -0.0001782, 0.0001724
3: -0.0022591, -0.0017397, -0.0022549, -0.0017399, -0.0001843, 0.0001783
4: -0.0021536, -0.0015913, -0.0021534, -0.0015958, -0.0001930, 0.0001995
5: 0.0056193, 0.0061514, 0.0056236, 0.0061512, -0.0001888, 0.0001827
6: -0.0000047, 0.0021066, 0.0000123, 0.0021057, -0.0007491, 0.0007248
7: -0.0054257, -0.0025503, -0.0054245, -0.0025735, -0.0009872, 0.0010202
8: 0.9853919, 0.9874175, 0.9853927, 0.9874010, -0.0006954, 0.0007187
9: -0.0044656, -0.0026270, -0.0044508, -0.0026278, -0.0006524, 0.0006312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005592
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005593
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033212, 0.0042231, 0.0033437, 0.0042289, -0.0003557, 0.0003266
1: 0.0018021, 0.0019324, 0.0018054, 0.0019332, -0.0000514, 0.0000472
2: 0.0120250, 0.0125236, 0.0120218, 0.0125112, -0.0001806, 0.0001967
3: -0.0022436, -0.0017279, -0.0022469, -0.0017408, -0.0001868, 0.0002034
4: -0.0021664, -0.0016081, -0.0021525, -0.0016046, -0.0002202, 0.0002022
5: 0.0056352, 0.0061635, 0.0056318, 0.0061503, -0.0001913, 0.0002084
6: 0.0000584, 0.0021546, 0.0000450, 0.0021024, -0.0007592, 0.0008267
7: -0.0054910, -0.0026363, -0.0054199, -0.0026180, -0.0011260, 0.0010340
8: 0.9853458, 0.9873568, 0.9853960, 0.9873697, -0.0007931, 0.0007284
9: -0.0044106, -0.0025852, -0.0044223, -0.0026307, -0.0006612, 0.0007200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005554
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005565
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033760, 0.0042944, 0.0033653, 0.0043160, -0.0004690, 0.0004591
1: 0.0018100, 0.0019427, 0.0018085, 0.0019458, -0.0000678, 0.0000663
2: 0.0119856, 0.0124933, 0.0119736, 0.0124993, -0.0002538, 0.0002593
3: -0.0022844, -0.0017593, -0.0022968, -0.0017531, -0.0002625, 0.0002682
4: -0.0021325, -0.0015640, -0.0021391, -0.0015506, -0.0002903, 0.0002842
5: 0.0055934, 0.0061314, 0.0055807, 0.0061377, -0.0002690, 0.0002748
6: -0.0001074, 0.0020272, -0.0001577, 0.0020521, -0.0010671, 0.0010902
7: -0.0053176, -0.0024105, -0.0053515, -0.0023420, -0.0014847, 0.0014533
8: 0.9854681, 0.9875159, 0.9854442, 0.9875642, -0.0010459, 0.0010238
9: -0.0045550, -0.0026962, -0.0045988, -0.0026744, -0.0009293, 0.0009494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005556, upper bound: 0.0005506
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005556, upper bound: 0.0005507
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033774, 0.0042800, 0.0033414, 0.0042916, -0.0004701, 0.0004941
1: 0.0018102, 0.0019406, 0.0018050, 0.0019423, -0.0000679, 0.0000714
2: 0.0119936, 0.0124926, 0.0119871, 0.0125125, -0.0002732, 0.0002599
3: -0.0022761, -0.0017600, -0.0022828, -0.0017395, -0.0002825, 0.0002688
4: -0.0021316, -0.0015729, -0.0021539, -0.0015657, -0.0002910, 0.0003059
5: 0.0056018, 0.0061306, 0.0055950, 0.0061517, -0.0002894, 0.0002754
6: -0.0000739, 0.0020240, -0.0001010, 0.0021077, -0.0011484, 0.0010927
7: -0.0053133, -0.0024561, -0.0054272, -0.0024192, -0.0014882, 0.0015640
8: 0.9854711, 0.9874837, 0.9853908, 0.9875097, -0.0010483, 0.0011017
9: -0.0045258, -0.0026989, -0.0045494, -0.0026261, -0.0010001, 0.0009516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005486, upper bound: 0.0005497
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005486, upper bound: 0.0005499
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033405, 0.0042583, 0.0033653, 0.0043160, -0.0004823, 0.0003991
1: 0.0018049, 0.0019375, 0.0018085, 0.0019458, -0.0000697, 0.0000577
2: 0.0120056, 0.0125130, 0.0119736, 0.0124993, -0.0002207, 0.0002667
3: -0.0022637, -0.0017389, -0.0022968, -0.0017531, -0.0002282, 0.0002758
4: -0.0021545, -0.0015863, -0.0021391, -0.0015506, -0.0002986, 0.0002471
5: 0.0056146, 0.0061522, 0.0055807, 0.0061377, -0.0002338, 0.0002826
6: -0.0000234, 0.0021098, -0.0001577, 0.0020521, -0.0009277, 0.0011211
7: -0.0054301, -0.0025248, -0.0053515, -0.0023420, -0.0015268, 0.0012634
8: 0.9853888, 0.9874353, 0.9854442, 0.9875642, -0.0010755, 0.0008900
9: -0.0044819, -0.0026242, -0.0045988, -0.0026744, -0.0008079, 0.0009763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005635, upper bound: 0.0005674
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005635, upper bound: 0.0005675
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033419, 0.0042442, 0.0033414, 0.0042916, -0.0004841, 0.0004304
1: 0.0018051, 0.0019355, 0.0018050, 0.0019423, -0.0000699, 0.0000622
2: 0.0120134, 0.0125122, 0.0119871, 0.0125125, -0.0002380, 0.0002676
3: -0.0022557, -0.0017397, -0.0022828, -0.0017395, -0.0002461, 0.0002768
4: -0.0021536, -0.0015951, -0.0021539, -0.0015657, -0.0002997, 0.0002664
5: 0.0056228, 0.0061514, 0.0055950, 0.0061517, -0.0002521, 0.0002836
6: 0.0000093, 0.0021065, -0.0001010, 0.0021077, -0.0010004, 0.0011251
7: -0.0054255, -0.0025694, -0.0054272, -0.0024192, -0.0015324, 0.0013625
8: 0.9853920, 0.9874039, 0.9853908, 0.9875097, -0.0010794, 0.0009598
9: -0.0044534, -0.0026271, -0.0045494, -0.0026261, -0.0008712, 0.0009798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005590, upper bound: 0.0005674
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005590, upper bound: 0.0005675
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033784, 0.0042675, 0.0033184, 0.0042138, -0.0003334, 0.0004918
1: 0.0018104, 0.0019388, 0.0018017, 0.0019311, -0.0000482, 0.0000711
2: 0.0120005, 0.0124920, 0.0120302, 0.0125252, -0.0002719, 0.0001843
3: -0.0022690, -0.0017606, -0.0022383, -0.0017263, -0.0002812, 0.0001906
4: -0.0021310, -0.0015806, -0.0021682, -0.0016139, -0.0002064, 0.0003045
5: 0.0056092, 0.0061300, 0.0056406, 0.0061652, -0.0002881, 0.0001953
6: -0.0000448, 0.0020217, 0.0000799, 0.0021612, -0.0011432, 0.0007748
7: -0.0053101, -0.0024956, -0.0055001, -0.0026656, -0.0010552, 0.0015569
8: 0.9854734, 0.9874558, 0.9853395, 0.9873362, -0.0007433, 0.0010967
9: -0.0045006, -0.0027009, -0.0043919, -0.0025794, -0.0009955, 0.0006747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005436, upper bound: 0.0005444
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005436, upper bound: 0.0005439
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033573, 0.0042407, 0.0033198, 0.0042002, -0.0003740, 0.0004932
1: 0.0018073, 0.0019350, 0.0018019, 0.0019291, -0.0000540, 0.0000713
2: 0.0120153, 0.0125037, 0.0120377, 0.0125244, -0.0002727, 0.0002068
3: -0.0022537, -0.0017486, -0.0022305, -0.0017271, -0.0002820, 0.0002139
4: -0.0021440, -0.0015972, -0.0021673, -0.0016223, -0.0002315, 0.0003053
5: 0.0056248, 0.0061423, 0.0056486, 0.0061643, -0.0002889, 0.0002191
6: 0.0000174, 0.0020706, 0.0001116, 0.0021579, -0.0011463, 0.0008693
7: -0.0053768, -0.0025804, -0.0054956, -0.0027087, -0.0011839, 0.0015612
8: 0.9854264, 0.9873961, 0.9853426, 0.9873058, -0.0008340, 0.0010997
9: -0.0044464, -0.0026583, -0.0043643, -0.0025823, -0.0009983, 0.0007570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005436, upper bound: 0.0005402
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005436, upper bound: 0.0005412
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033429, 0.0042340, 0.0033184, 0.0042138, -0.0003072, 0.0004098
1: 0.0018052, 0.0019340, 0.0018017, 0.0019311, -0.0000444, 0.0000592
2: 0.0120190, 0.0125117, 0.0120302, 0.0125252, -0.0002266, 0.0001698
3: -0.0022498, -0.0017403, -0.0022383, -0.0017263, -0.0002343, 0.0001757
4: -0.0021530, -0.0016014, -0.0021682, -0.0016139, -0.0001902, 0.0002537
5: 0.0056288, 0.0061508, 0.0056406, 0.0061652, -0.0002401, 0.0001800
6: 0.0000331, 0.0021042, 0.0000799, 0.0021612, -0.0009525, 0.0007140
7: -0.0054225, -0.0026018, -0.0055001, -0.0026656, -0.0009724, 0.0012972
8: 0.9853941, 0.9873811, 0.9853395, 0.9873362, -0.0006850, 0.0009138
9: -0.0044327, -0.0026290, -0.0043919, -0.0025794, -0.0008295, 0.0006218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005445
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005440, upper bound: 0.0005447
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033222, 0.0042070, 0.0033198, 0.0042002, -0.0003508, 0.0004120
1: 0.0018023, 0.0019301, 0.0018019, 0.0019291, -0.0000507, 0.0000595
2: 0.0120339, 0.0125231, 0.0120377, 0.0125244, -0.0002278, 0.0001939
3: -0.0022344, -0.0017285, -0.0022305, -0.0017271, -0.0002356, 0.0002006
4: -0.0021658, -0.0016181, -0.0021673, -0.0016223, -0.0002171, 0.0002550
5: 0.0056446, 0.0061629, 0.0056486, 0.0061643, -0.0002413, 0.0002055
6: 0.0000958, 0.0021524, 0.0001116, 0.0021579, -0.0009576, 0.0008153
7: -0.0054881, -0.0026872, -0.0054956, -0.0027087, -0.0011104, 0.0013041
8: 0.9853480, 0.9873210, 0.9853426, 0.9873058, -0.0007822, 0.0009187
9: -0.0043781, -0.0025871, -0.0043643, -0.0025823, -0.0008339, 0.0007100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005407
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005440, upper bound: 0.0005429
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033770, 0.0042754, 0.0033421, 0.0042864, -0.0004626, 0.0005187
1: 0.0018102, 0.0019400, 0.0018051, 0.0019416, -0.0000668, 0.0000749
2: 0.0119961, 0.0124928, 0.0119900, 0.0125121, -0.0002868, 0.0002558
3: -0.0022735, -0.0017598, -0.0022798, -0.0017399, -0.0002966, 0.0002645
4: -0.0021318, -0.0015757, -0.0021535, -0.0015689, -0.0002864, 0.0003211
5: 0.0056045, 0.0061308, 0.0055981, 0.0061513, -0.0003039, 0.0002710
6: -0.0000633, 0.0020249, -0.0000889, 0.0021061, -0.0012057, 0.0010752
7: -0.0053144, -0.0024705, -0.0054250, -0.0024356, -0.0014643, 0.0016420
8: 0.9854703, 0.9874735, 0.9853923, 0.9874982, -0.0010315, 0.0011567
9: -0.0045167, -0.0026982, -0.0045389, -0.0026275, -0.0010500, 0.0009363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005391
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005381
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033784, 0.0042613, 0.0033157, 0.0042647, -0.0004646, 0.0005447
1: 0.0018104, 0.0019379, 0.0018013, 0.0019384, -0.0000671, 0.0000787
2: 0.0120039, 0.0124921, 0.0120020, 0.0125267, -0.0003011, 0.0002569
3: -0.0022655, -0.0017606, -0.0022674, -0.0017248, -0.0003115, 0.0002656
4: -0.0021310, -0.0015845, -0.0021698, -0.0015824, -0.0002876, 0.0003372
5: 0.0056128, 0.0061300, 0.0056108, 0.0061667, -0.0003191, 0.0002721
6: -0.0000305, 0.0020217, -0.0000382, 0.0021674, -0.0012660, 0.0010798
7: -0.0053101, -0.0025152, -0.0055085, -0.0025046, -0.0014706, 0.0017242
8: 0.9854733, 0.9874421, 0.9853336, 0.9874496, -0.0010359, 0.0012145
9: -0.0044880, -0.0027009, -0.0044948, -0.0025741, -0.0011025, 0.0009403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005287, upper bound: 0.0005371
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005307, upper bound: 0.0005371
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033415, 0.0042422, 0.0033421, 0.0042864, -0.0004770, 0.0004485
1: 0.0018051, 0.0019352, 0.0018051, 0.0019416, -0.0000689, 0.0000648
2: 0.0120145, 0.0125124, 0.0119900, 0.0125121, -0.0002480, 0.0002637
3: -0.0022545, -0.0017395, -0.0022798, -0.0017399, -0.0002564, 0.0002727
4: -0.0021538, -0.0015963, -0.0021535, -0.0015689, -0.0002952, 0.0002776
5: 0.0056240, 0.0061516, 0.0055981, 0.0061513, -0.0002627, 0.0002794
6: 0.0000141, 0.0021074, -0.0000889, 0.0021061, -0.0010424, 0.0011086
7: -0.0054268, -0.0025759, -0.0054250, -0.0024356, -0.0015098, 0.0014196
8: 0.9853911, 0.9873994, 0.9853923, 0.9874982, -0.0010635, 0.0010000
9: -0.0044493, -0.0026263, -0.0045389, -0.0026275, -0.0009078, 0.0009654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005521
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005452, upper bound: 0.0005546
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033429, 0.0042280, 0.0033157, 0.0042647, -0.0004792, 0.0004780
1: 0.0018053, 0.0019331, 0.0018013, 0.0019384, -0.0000692, 0.0000691
2: 0.0120223, 0.0125116, 0.0120020, 0.0125267, -0.0002643, 0.0002650
3: -0.0022464, -0.0017403, -0.0022674, -0.0017248, -0.0002734, 0.0002740
4: -0.0021529, -0.0016051, -0.0021698, -0.0015824, -0.0002967, 0.0002959
5: 0.0056323, 0.0061508, 0.0056108, 0.0061667, -0.0002800, 0.0002807
6: 0.0000469, 0.0021041, -0.0000382, 0.0021674, -0.0011111, 0.0011139
7: -0.0054223, -0.0026206, -0.0055085, -0.0025046, -0.0015170, 0.0015132
8: 0.9853943, 0.9873679, 0.9853336, 0.9874496, -0.0010686, 0.0010660
9: -0.0044207, -0.0026292, -0.0044948, -0.0025741, -0.0009676, 0.0009700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005421, upper bound: 0.0005521
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005422, upper bound: 0.0005546
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033693, 0.0043160, 0.0033778, 0.0042796, -0.0004187, 0.0004675
1: 0.0018091, 0.0019458, 0.0018103, 0.0019406, -0.0000605, 0.0000675
2: 0.0119737, 0.0124971, 0.0119938, 0.0124924, -0.0002584, 0.0002315
3: -0.0022967, -0.0017554, -0.0022759, -0.0017603, -0.0002673, 0.0002394
4: -0.0021366, -0.0015506, -0.0021314, -0.0015731, -0.0002592, 0.0002894
5: 0.0055807, 0.0061353, 0.0056021, 0.0061303, -0.0002738, 0.0002452
6: -0.0001576, 0.0020429, -0.0000729, 0.0020231, -0.0010865, 0.0009731
7: -0.0053389, -0.0023421, -0.0053120, -0.0024574, -0.0013252, 0.0014797
8: 0.9854531, 0.9875641, 0.9854720, 0.9874828, -0.0009335, 0.0010423
9: -0.0045987, -0.0026825, -0.0045250, -0.0026997, -0.0009462, 0.0008474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005355
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005420, upper bound: 0.0005359
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033478, 0.0042864, 0.0033778, 0.0042796, -0.0004644, 0.0004622
1: 0.0018060, 0.0019416, 0.0018103, 0.0019406, -0.0000671, 0.0000668
2: 0.0119900, 0.0125089, 0.0119938, 0.0124924, -0.0002555, 0.0002568
3: -0.0022798, -0.0017431, -0.0022759, -0.0017603, -0.0002643, 0.0002656
4: -0.0021499, -0.0015689, -0.0021314, -0.0015731, -0.0002875, 0.0002861
5: 0.0055981, 0.0061479, 0.0056021, 0.0061303, -0.0002707, 0.0002721
6: -0.0000887, 0.0020927, -0.0000729, 0.0020231, -0.0010743, 0.0010795
7: -0.0054068, -0.0024358, -0.0053120, -0.0024574, -0.0014701, 0.0014630
8: 0.9854051, 0.9874980, 0.9854720, 0.9874828, -0.0010356, 0.0010306
9: -0.0045388, -0.0026391, -0.0045250, -0.0026997, -0.0009355, 0.0009400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005359
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005420, upper bound: 0.0005359
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033455, 0.0042916, 0.0033791, 0.0042651, -0.0004534, 0.0004685
1: 0.0018056, 0.0019423, 0.0018105, 0.0019385, -0.0000655, 0.0000677
2: 0.0119871, 0.0125102, 0.0120018, 0.0124916, -0.0002590, 0.0002506
3: -0.0022828, -0.0017418, -0.0022676, -0.0017610, -0.0002679, 0.0002592
4: -0.0021513, -0.0015657, -0.0021305, -0.0015821, -0.0002806, 0.0002900
5: 0.0055950, 0.0061492, 0.0056106, 0.0061296, -0.0002745, 0.0002656
6: -0.0001009, 0.0020980, -0.0000393, 0.0020200, -0.0010890, 0.0010537
7: -0.0054141, -0.0024193, -0.0053077, -0.0025032, -0.0014351, 0.0014832
8: 0.9854001, 0.9875096, 0.9854749, 0.9874505, -0.0010109, 0.0010448
9: -0.0045494, -0.0026345, -0.0044958, -0.0027024, -0.0009484, 0.0009176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005286
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005306
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033220, 0.0042646, 0.0033791, 0.0042651, -0.0004975, 0.0004625
1: 0.0018022, 0.0019384, 0.0018105, 0.0019385, -0.0000719, 0.0000668
2: 0.0120021, 0.0125232, 0.0120018, 0.0124916, -0.0002557, 0.0002751
3: -0.0022674, -0.0017284, -0.0022676, -0.0017610, -0.0002644, 0.0002845
4: -0.0021659, -0.0015824, -0.0021305, -0.0015821, -0.0003080, 0.0002863
5: 0.0056108, 0.0061630, 0.0056106, 0.0061296, -0.0002709, 0.0002915
6: -0.0000381, 0.0021528, -0.0000393, 0.0020200, -0.0010749, 0.0011564
7: -0.0054886, -0.0025048, -0.0053077, -0.0025032, -0.0015750, 0.0014639
8: 0.9853476, 0.9874494, 0.9854749, 0.9874505, -0.0011094, 0.0010312
9: -0.0044947, -0.0025868, -0.0044958, -0.0027024, -0.0009361, 0.0010071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005287
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005307
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033693, 0.0043160, 0.0034069, 0.0043532, -0.0003718, 0.0003352
1: 0.0018091, 0.0019458, 0.0018145, 0.0019512, -0.0000537, 0.0000484
2: 0.0119737, 0.0124971, 0.0119531, 0.0124762, -0.0001853, 0.0002056
3: -0.0022967, -0.0017554, -0.0023180, -0.0017769, -0.0001917, 0.0002126
4: -0.0021366, -0.0015506, -0.0021133, -0.0015276, -0.0002302, 0.0002075
5: 0.0055807, 0.0061353, 0.0055590, 0.0061133, -0.0001964, 0.0002178
6: -0.0001576, 0.0020429, -0.0002440, 0.0019553, -0.0007792, 0.0008642
7: -0.0053389, -0.0023421, -0.0052197, -0.0022243, -0.0011770, 0.0010612
8: 0.9854531, 0.9875641, 0.9855370, 0.9876471, -0.0008291, 0.0007475
9: -0.0045987, -0.0026825, -0.0046740, -0.0027587, -0.0006785, 0.0007526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005289
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005420, upper bound: 0.0005285
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033478, 0.0042864, 0.0034069, 0.0043532, -0.0004461, 0.0003467
1: 0.0018060, 0.0019416, 0.0018145, 0.0019512, -0.0000644, 0.0000501
2: 0.0119900, 0.0125089, 0.0119531, 0.0124762, -0.0001917, 0.0002466
3: -0.0022798, -0.0017431, -0.0023180, -0.0017769, -0.0001983, 0.0002551
4: -0.0021499, -0.0015689, -0.0021133, -0.0015276, -0.0002761, 0.0002146
5: 0.0055981, 0.0061479, 0.0055590, 0.0061133, -0.0002031, 0.0002613
6: -0.0000887, 0.0020927, -0.0002440, 0.0019553, -0.0008059, 0.0010368
7: -0.0054068, -0.0024358, -0.0052197, -0.0022243, -0.0014120, 0.0010976
8: 0.9854051, 0.9874980, 0.9855370, 0.9876471, -0.0009947, 0.0007731
9: -0.0045388, -0.0026391, -0.0046740, -0.0027587, -0.0007018, 0.0009029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005285
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005420, upper bound: 0.0005285
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033455, 0.0042916, 0.0034084, 0.0043403, -0.0004130, 0.0003379
1: 0.0018056, 0.0019423, 0.0018147, 0.0019494, -0.0000597, 0.0000488
2: 0.0119871, 0.0125102, 0.0119602, 0.0124754, -0.0001868, 0.0002283
3: -0.0022828, -0.0017418, -0.0023106, -0.0017778, -0.0001932, 0.0002361
4: -0.0021513, -0.0015657, -0.0021124, -0.0015355, -0.0002556, 0.0002092
5: 0.0055950, 0.0061492, 0.0055665, 0.0061124, -0.0001980, 0.0002419
6: -0.0001009, 0.0020980, -0.0002141, 0.0019519, -0.0007854, 0.0009599
7: -0.0054141, -0.0024193, -0.0052150, -0.0022651, -0.0013073, 0.0010697
8: 0.9854001, 0.9875096, 0.9855403, 0.9876182, -0.0009209, 0.0007535
9: -0.0045494, -0.0026345, -0.0046480, -0.0027617, -0.0006840, 0.0008359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005202
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005207
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033220, 0.0042646, 0.0034084, 0.0043403, -0.0004736, 0.0003487
1: 0.0018022, 0.0019384, 0.0018147, 0.0019494, -0.0000684, 0.0000504
2: 0.0120021, 0.0125232, 0.0119602, 0.0124754, -0.0001928, 0.0002619
3: -0.0022674, -0.0017284, -0.0023106, -0.0017778, -0.0001994, 0.0002708
4: -0.0021659, -0.0015824, -0.0021124, -0.0015355, -0.0002932, 0.0002158
5: 0.0056108, 0.0061630, 0.0055665, 0.0061124, -0.0002042, 0.0002775
6: -0.0000381, 0.0021528, -0.0002141, 0.0019519, -0.0008104, 0.0011009
7: -0.0054886, -0.0025048, -0.0052150, -0.0022651, -0.0014993, 0.0011036
8: 0.9853476, 0.9874494, 0.9855403, 0.9876182, -0.0010561, 0.0007774
9: -0.0044947, -0.0025868, -0.0046480, -0.0027617, -0.0007057, 0.0009587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005204
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005210
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033698, 0.0043033, 0.0033572, 0.0042401, -0.0004011, 0.0005424
1: 0.0018091, 0.0019440, 0.0018073, 0.0019349, -0.0000580, 0.0000784
2: 0.0119807, 0.0124968, 0.0120156, 0.0125038, -0.0002999, 0.0002218
3: -0.0022895, -0.0017557, -0.0022533, -0.0017485, -0.0003101, 0.0002294
4: -0.0021363, -0.0015585, -0.0021441, -0.0015976, -0.0002483, 0.0003358
5: 0.0055882, 0.0061350, 0.0056252, 0.0061424, -0.0003177, 0.0002350
6: -0.0001280, 0.0020416, 0.0000190, 0.0020710, -0.0012607, 0.0009324
7: -0.0053371, -0.0023823, -0.0053772, -0.0025825, -0.0012698, 0.0017169
8: 0.9854543, 0.9875357, 0.9854261, 0.9873947, -0.0008945, 0.0012095
9: -0.0045730, -0.0026836, -0.0044450, -0.0026580, -0.0010979, 0.0008119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005355
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005355
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033671, 0.0043011, 0.0033571, 0.0042396, -0.0004107, 0.0005463
1: 0.0018087, 0.0019437, 0.0018073, 0.0019348, -0.0000593, 0.0000789
2: 0.0119819, 0.0124983, 0.0120159, 0.0125038, -0.0003021, 0.0002270
3: -0.0022882, -0.0017541, -0.0022530, -0.0017484, -0.0003124, 0.0002348
4: -0.0021380, -0.0015598, -0.0021442, -0.0015979, -0.0002542, 0.0003382
5: 0.0055895, 0.0061366, 0.0056255, 0.0061425, -0.0003200, 0.0002406
6: -0.0001230, 0.0020480, 0.0000201, 0.0020711, -0.0012698, 0.0009545
7: -0.0053459, -0.0023892, -0.0053774, -0.0025841, -0.0012999, 0.0017294
8: 0.9854481, 0.9875309, 0.9854259, 0.9873935, -0.0009157, 0.0012182
9: -0.0045686, -0.0026781, -0.0044440, -0.0026579, -0.0011058, 0.0008312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005359
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005359
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033462, 0.0042787, 0.0033586, 0.0042264, -0.0004362, 0.0005426
1: 0.0018057, 0.0019404, 0.0018075, 0.0019329, -0.0000630, 0.0000784
2: 0.0119943, 0.0125098, 0.0120232, 0.0125030, -0.0003000, 0.0002412
3: -0.0022754, -0.0017422, -0.0022455, -0.0017493, -0.0003102, 0.0002494
4: -0.0021509, -0.0015737, -0.0021433, -0.0016061, -0.0002700, 0.0003359
5: 0.0056026, 0.0061489, 0.0056332, 0.0061416, -0.0003178, 0.0002556
6: -0.0000708, 0.0020965, 0.0000507, 0.0020678, -0.0012611, 0.0010140
7: -0.0054120, -0.0024602, -0.0053729, -0.0026258, -0.0013809, 0.0017175
8: 0.9854016, 0.9874808, 0.9854292, 0.9873642, -0.0009727, 0.0012098
9: -0.0045232, -0.0026358, -0.0044173, -0.0026608, -0.0010982, 0.0008830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005287
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005287
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033424, 0.0042779, 0.0033585, 0.0042264, -0.0004460, 0.0005472
1: 0.0018052, 0.0019403, 0.0018075, 0.0019329, -0.0000644, 0.0000791
2: 0.0119947, 0.0125119, 0.0120232, 0.0125030, -0.0003026, 0.0002466
3: -0.0022749, -0.0017400, -0.0022455, -0.0017492, -0.0003129, 0.0002550
4: -0.0021533, -0.0015742, -0.0021433, -0.0016061, -0.0002761, 0.0003387
5: 0.0056031, 0.0061511, 0.0056332, 0.0061417, -0.0003206, 0.0002613
6: -0.0000690, 0.0021054, 0.0000507, 0.0020680, -0.0012719, 0.0010366
7: -0.0054240, -0.0024628, -0.0053731, -0.0026258, -0.0014117, 0.0017323
8: 0.9853932, 0.9874791, 0.9854289, 0.9873642, -0.0009944, 0.0012202
9: -0.0045216, -0.0026281, -0.0044174, -0.0026606, -0.0011077, 0.0009027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005307
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005307
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033698, 0.0043033, 0.0033866, 0.0043148, -0.0003567, 0.0004214
1: 0.0018091, 0.0019440, 0.0018116, 0.0019457, -0.0000515, 0.0000609
2: 0.0119807, 0.0124968, 0.0119743, 0.0124875, -0.0002330, 0.0001972
3: -0.0022895, -0.0017557, -0.0022960, -0.0017653, -0.0002410, 0.0002039
4: -0.0021363, -0.0015585, -0.0021259, -0.0015514, -0.0002208, 0.0002609
5: 0.0055882, 0.0061350, 0.0055815, 0.0061252, -0.0002469, 0.0002089
6: -0.0001280, 0.0020416, -0.0001547, 0.0020026, -0.0009796, 0.0008290
7: -0.0053371, -0.0023823, -0.0052840, -0.0023461, -0.0011290, 0.0013341
8: 0.9854543, 0.9875357, 0.9854917, 0.9875613, -0.0007953, 0.0009397
9: -0.0045730, -0.0026836, -0.0045962, -0.0027176, -0.0008530, 0.0007219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005289
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005289
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033671, 0.0043011, 0.0033865, 0.0043143, -0.0003689, 0.0004261
1: 0.0018087, 0.0019437, 0.0018116, 0.0019456, -0.0000533, 0.0000616
2: 0.0119819, 0.0124983, 0.0119746, 0.0124875, -0.0002356, 0.0002039
3: -0.0022882, -0.0017541, -0.0022958, -0.0017653, -0.0002436, 0.0002109
4: -0.0021380, -0.0015598, -0.0021260, -0.0015517, -0.0002283, 0.0002638
5: 0.0055895, 0.0061366, 0.0055818, 0.0061252, -0.0002496, 0.0002161
6: -0.0001230, 0.0020480, -0.0001536, 0.0020028, -0.0009903, 0.0008574
7: -0.0053459, -0.0023892, -0.0052843, -0.0023476, -0.0011677, 0.0013488
8: 0.9854481, 0.9875309, 0.9854915, 0.9875602, -0.0008226, 0.0009501
9: -0.0045686, -0.0026781, -0.0045952, -0.0027174, -0.0008624, 0.0007467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005285
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005285
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033462, 0.0042787, 0.0033881, 0.0043027, -0.0003975, 0.0004232
1: 0.0018057, 0.0019404, 0.0018118, 0.0019439, -0.0000574, 0.0000611
2: 0.0119943, 0.0125098, 0.0119810, 0.0124867, -0.0002340, 0.0002198
3: -0.0022754, -0.0017422, -0.0022891, -0.0017661, -0.0002420, 0.0002273
4: -0.0021509, -0.0015737, -0.0021250, -0.0015588, -0.0002461, 0.0002619
5: 0.0056026, 0.0061489, 0.0055885, 0.0061243, -0.0002479, 0.0002329
6: -0.0000708, 0.0020965, -0.0001267, 0.0019992, -0.0009835, 0.0009240
7: -0.0054120, -0.0024602, -0.0052794, -0.0023842, -0.0012584, 0.0013395
8: 0.9854016, 0.9874808, 0.9854950, 0.9875344, -0.0008864, 0.0009436
9: -0.0045232, -0.0026358, -0.0045718, -0.0027205, -0.0008565, 0.0008046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005204
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005204
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033424, 0.0042779, 0.0033880, 0.0043021, -0.0004083, 0.0004281
1: 0.0018052, 0.0019403, 0.0018118, 0.0019438, -0.0000590, 0.0000619
2: 0.0119947, 0.0125119, 0.0119813, 0.0124867, -0.0002367, 0.0002257
3: -0.0022749, -0.0017400, -0.0022888, -0.0017661, -0.0002448, 0.0002335
4: -0.0021533, -0.0015742, -0.0021251, -0.0015592, -0.0002527, 0.0002650
5: 0.0056031, 0.0061511, 0.0055889, 0.0061244, -0.0002508, 0.0002392
6: -0.0000690, 0.0021054, -0.0001253, 0.0019994, -0.0009951, 0.0009490
7: -0.0054240, -0.0024628, -0.0052798, -0.0023860, -0.0012925, 0.0013552
8: 0.9853932, 0.9874791, 0.9854947, 0.9875332, -0.0009104, 0.0009547
9: -0.0045216, -0.0026281, -0.0045707, -0.0027203, -0.0008666, 0.0008264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005210
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005210
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0034085, 0.0043444, 0.0033405, 0.0042583, -0.0004256, 0.0005775
1: 0.0018147, 0.0019499, 0.0018049, 0.0019375, -0.0000615, 0.0000834
2: 0.0119579, 0.0124754, 0.0120056, 0.0125130, -0.0003193, 0.0002353
3: -0.0023130, -0.0017778, -0.0022637, -0.0017389, -0.0003302, 0.0002434
4: -0.0021124, -0.0015330, -0.0021545, -0.0015863, -0.0002635, 0.0003575
5: 0.0055641, 0.0061124, 0.0056146, 0.0061522, -0.0003383, 0.0002493
6: -0.0002236, 0.0019518, -0.0000234, 0.0021098, -0.0013424, 0.0009893
7: -0.0052149, -0.0022522, -0.0054301, -0.0025248, -0.0013474, 0.0018282
8: 0.9855403, 0.9876274, 0.9853888, 0.9874353, -0.0009491, 0.0012878
9: -0.0046563, -0.0027618, -0.0044819, -0.0026242, -0.0011690, 0.0008615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005568
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005568
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033846, 0.0043203, 0.0033419, 0.0042442, -0.0004534, 0.0005784
1: 0.0018113, 0.0019465, 0.0018051, 0.0019355, -0.0000655, 0.0000836
2: 0.0119713, 0.0124886, 0.0120134, 0.0125122, -0.0003198, 0.0002507
3: -0.0022992, -0.0017641, -0.0022557, -0.0017397, -0.0003307, 0.0002592
4: -0.0021272, -0.0015479, -0.0021536, -0.0015951, -0.0002806, 0.0003580
5: 0.0055782, 0.0061264, 0.0056228, 0.0061514, -0.0003388, 0.0002656
6: -0.0001675, 0.0020073, 0.0000093, 0.0021065, -0.0013443, 0.0010538
7: -0.0052905, -0.0023285, -0.0054255, -0.0025694, -0.0014351, 0.0018308
8: 0.9854872, 0.9875736, 0.9853920, 0.9874039, -0.0010109, 0.0012897
9: -0.0046074, -0.0027135, -0.0044534, -0.0026271, -0.0011707, 0.0009177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005485
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005485
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033653, 0.0043160, 0.0033405, 0.0042583, -0.0003991, 0.0004823
1: 0.0018085, 0.0019458, 0.0018049, 0.0019375, -0.0000577, 0.0000697
2: 0.0119736, 0.0124993, 0.0120056, 0.0125130, -0.0002667, 0.0002207
3: -0.0022968, -0.0017531, -0.0022637, -0.0017389, -0.0002758, 0.0002282
4: -0.0021391, -0.0015506, -0.0021545, -0.0015863, -0.0002471, 0.0002986
5: 0.0055807, 0.0061377, 0.0056146, 0.0061522, -0.0002826, 0.0002338
6: -0.0001577, 0.0020521, -0.0000234, 0.0021098, -0.0011211, 0.0009277
7: -0.0053515, -0.0023420, -0.0054301, -0.0025248, -0.0012634, 0.0015268
8: 0.9854442, 0.9875642, 0.9853888, 0.9874353, -0.0008900, 0.0010755
9: -0.0045988, -0.0026744, -0.0044819, -0.0026242, -0.0009763, 0.0008079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005624
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005624
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033414, 0.0042916, 0.0033419, 0.0042442, -0.0004304, 0.0004841
1: 0.0018050, 0.0019423, 0.0018051, 0.0019355, -0.0000622, 0.0000699
2: 0.0119871, 0.0125125, 0.0120134, 0.0125122, -0.0002676, 0.0002380
3: -0.0022828, -0.0017395, -0.0022557, -0.0017397, -0.0002768, 0.0002461
4: -0.0021539, -0.0015657, -0.0021536, -0.0015951, -0.0002664, 0.0002997
5: 0.0055950, 0.0061517, 0.0056228, 0.0061514, -0.0002836, 0.0002521
6: -0.0001010, 0.0021077, 0.0000093, 0.0021065, -0.0011251, 0.0010004
7: -0.0054272, -0.0024192, -0.0054255, -0.0025694, -0.0013625, 0.0015324
8: 0.9853908, 0.9875097, 0.9853920, 0.9874039, -0.0009598, 0.0010794
9: -0.0045494, -0.0026261, -0.0044534, -0.0026271, -0.0009798, 0.0008712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005575
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005575
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034069, 0.0043532, 0.0033637, 0.0043318, -0.0003563, 0.0004151
1: 0.0018145, 0.0019512, 0.0018083, 0.0019481, -0.0000515, 0.0000600
2: 0.0119531, 0.0124762, 0.0119649, 0.0125002, -0.0002295, 0.0001970
3: -0.0023180, -0.0017769, -0.0023058, -0.0017522, -0.0002374, 0.0002037
4: -0.0021133, -0.0015276, -0.0021401, -0.0015408, -0.0002205, 0.0002570
5: 0.0055590, 0.0061133, 0.0055715, 0.0061386, -0.0002432, 0.0002087
6: -0.0002440, 0.0019553, -0.0001944, 0.0020559, -0.0009649, 0.0008281
7: -0.0052197, -0.0022243, -0.0053567, -0.0022920, -0.0011278, 0.0013141
8: 0.9855370, 0.9876471, 0.9854405, 0.9875993, -0.0007945, 0.0009257
9: -0.0046740, -0.0027587, -0.0046308, -0.0026711, -0.0008403, 0.0007212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005437
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005437
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034084, 0.0043403, 0.0033398, 0.0043068, -0.0003584, 0.0004523
1: 0.0018147, 0.0019494, 0.0018048, 0.0019445, -0.0000518, 0.0000653
2: 0.0119602, 0.0124754, 0.0119787, 0.0125134, -0.0002500, 0.0001981
3: -0.0023106, -0.0017778, -0.0022915, -0.0017386, -0.0002586, 0.0002049
4: -0.0021124, -0.0015355, -0.0021549, -0.0015563, -0.0002218, 0.0002800
5: 0.0055665, 0.0061124, 0.0055861, 0.0061526, -0.0002649, 0.0002099
6: -0.0002141, 0.0019519, -0.0001362, 0.0021113, -0.0010512, 0.0008330
7: -0.0052150, -0.0022651, -0.0054322, -0.0023712, -0.0011345, 0.0014316
8: 0.9855403, 0.9876182, 0.9853873, 0.9875435, -0.0007991, 0.0010085
9: -0.0046480, -0.0027617, -0.0045801, -0.0026229, -0.0009154, 0.0007254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005437
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005437
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033638, 0.0043243, 0.0033637, 0.0043318, -0.0003352, 0.0003246
1: 0.0018083, 0.0019470, 0.0018083, 0.0019481, -0.0000484, 0.0000469
2: 0.0119690, 0.0125001, 0.0119649, 0.0125002, -0.0001795, 0.0001853
3: -0.0023015, -0.0017523, -0.0023058, -0.0017522, -0.0001856, 0.0001917
4: -0.0021400, -0.0015454, -0.0021401, -0.0015408, -0.0002075, 0.0002009
5: 0.0055759, 0.0061385, 0.0055715, 0.0061386, -0.0001901, 0.0001963
6: -0.0001770, 0.0020556, -0.0001944, 0.0020559, -0.0007544, 0.0007790
7: -0.0053562, -0.0023157, -0.0053567, -0.0022920, -0.0010610, 0.0010275
8: 0.9854409, 0.9875827, 0.9854405, 0.9875993, -0.0007474, 0.0007238
9: -0.0046156, -0.0026714, -0.0046308, -0.0026711, -0.0006570, 0.0006784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005551
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005551
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033653, 0.0043113, 0.0033398, 0.0043068, -0.0003386, 0.0003687
1: 0.0018085, 0.0019452, 0.0018048, 0.0019445, -0.0000489, 0.0000533
2: 0.0119763, 0.0124993, 0.0119787, 0.0125134, -0.0002038, 0.0001872
3: -0.0022940, -0.0017531, -0.0022915, -0.0017386, -0.0002108, 0.0001936
4: -0.0021391, -0.0015535, -0.0021549, -0.0015563, -0.0002096, 0.0002282
5: 0.0055835, 0.0061377, 0.0055861, 0.0061526, -0.0002160, 0.0001984
6: -0.0001466, 0.0020521, -0.0001362, 0.0021113, -0.0008569, 0.0007870
7: -0.0053515, -0.0023570, -0.0054322, -0.0023712, -0.0010719, 0.0011670
8: 0.9854442, 0.9875535, 0.9853873, 0.9875435, -0.0007551, 0.0008220
9: -0.0045892, -0.0026745, -0.0045801, -0.0026229, -0.0007462, 0.0006854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005551
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005551
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033877, 0.0043118, 0.0033415, 0.0042422, -0.0004658, 0.0005710
1: 0.0018117, 0.0019452, 0.0018051, 0.0019352, -0.0000673, 0.0000825
2: 0.0119760, 0.0124869, 0.0120145, 0.0125124, -0.0003157, 0.0002575
3: -0.0022944, -0.0017659, -0.0022545, -0.0017395, -0.0003265, 0.0002664
4: -0.0021252, -0.0015532, -0.0021538, -0.0015963, -0.0002884, 0.0003535
5: 0.0055832, 0.0061245, 0.0056240, 0.0061516, -0.0003345, 0.0002729
6: -0.0001479, 0.0020000, 0.0000141, 0.0021074, -0.0013272, 0.0010827
7: -0.0052806, -0.0023553, -0.0054268, -0.0025759, -0.0014746, 0.0018076
8: 0.9854941, 0.9875548, 0.9853911, 0.9873994, -0.0010387, 0.0012733
9: -0.0045903, -0.0027198, -0.0044493, -0.0026263, -0.0011558, 0.0009429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005265, upper bound: 0.0005378
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005379
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033620, 0.0042892, 0.0033429, 0.0042280, -0.0004926, 0.0005725
1: 0.0018080, 0.0019420, 0.0018053, 0.0019331, -0.0000712, 0.0000827
2: 0.0119885, 0.0125011, 0.0120223, 0.0125116, -0.0003165, 0.0002723
3: -0.0022814, -0.0017512, -0.0022464, -0.0017403, -0.0003273, 0.0002817
4: -0.0021411, -0.0015672, -0.0021529, -0.0016051, -0.0003049, 0.0003544
5: 0.0055964, 0.0061396, 0.0056323, 0.0061508, -0.0003353, 0.0002886
6: -0.0000954, 0.0020598, 0.0000469, 0.0021041, -0.0013306, 0.0011449
7: -0.0053620, -0.0024268, -0.0054223, -0.0026206, -0.0015593, 0.0018121
8: 0.9854367, 0.9875044, 0.9853943, 0.9873679, -0.0010984, 0.0012765
9: -0.0045446, -0.0026678, -0.0044207, -0.0026292, -0.0011587, 0.0009971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005256, upper bound: 0.0005325
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005325
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033421, 0.0042864, 0.0033415, 0.0042422, -0.0004485, 0.0004770
1: 0.0018051, 0.0019416, 0.0018051, 0.0019352, -0.0000648, 0.0000689
2: 0.0119900, 0.0125121, 0.0120145, 0.0125124, -0.0002637, 0.0002480
3: -0.0022798, -0.0017399, -0.0022545, -0.0017395, -0.0002727, 0.0002564
4: -0.0021535, -0.0015689, -0.0021538, -0.0015963, -0.0002776, 0.0002952
5: 0.0055981, 0.0061513, 0.0056240, 0.0061516, -0.0002794, 0.0002627
6: -0.0000889, 0.0021061, 0.0000141, 0.0021074, -0.0011086, 0.0010424
7: -0.0054250, -0.0024356, -0.0054268, -0.0025759, -0.0014196, 0.0015098
8: 0.9853923, 0.9874982, 0.9853911, 0.9873994, -0.0010000, 0.0010635
9: -0.0045389, -0.0026275, -0.0044493, -0.0026263, -0.0009654, 0.0009078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005437
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005438
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033157, 0.0042647, 0.0033429, 0.0042280, -0.0004780, 0.0004792
1: 0.0018013, 0.0019384, 0.0018053, 0.0019331, -0.0000691, 0.0000692
2: 0.0120020, 0.0125267, 0.0120223, 0.0125116, -0.0002650, 0.0002643
3: -0.0022674, -0.0017248, -0.0022464, -0.0017403, -0.0002740, 0.0002734
4: -0.0021698, -0.0015824, -0.0021529, -0.0016051, -0.0002959, 0.0002967
5: 0.0056108, 0.0061667, 0.0056323, 0.0061508, -0.0002807, 0.0002800
6: -0.0000382, 0.0021674, 0.0000469, 0.0021041, -0.0011139, 0.0011111
7: -0.0055085, -0.0025046, -0.0054223, -0.0026206, -0.0015132, 0.0015170
8: 0.9853336, 0.9874496, 0.9853943, 0.9873679, -0.0010660, 0.0010686
9: -0.0044948, -0.0025741, -0.0044207, -0.0026292, -0.0009700, 0.0009676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005413
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005414
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033862, 0.0043201, 0.0033648, 0.0043156, -0.0004370, 0.0004104
1: 0.0018115, 0.0019464, 0.0018084, 0.0019458, -0.0000631, 0.0000593
2: 0.0119714, 0.0124877, 0.0119739, 0.0124995, -0.0002269, 0.0002416
3: -0.0022991, -0.0017651, -0.0022965, -0.0017528, -0.0002346, 0.0002499
4: -0.0021261, -0.0015480, -0.0021394, -0.0015508, -0.0002705, 0.0002540
5: 0.0055783, 0.0061254, 0.0055810, 0.0061380, -0.0002404, 0.0002560
6: -0.0001672, 0.0020035, -0.0001567, 0.0020533, -0.0009538, 0.0010158
7: -0.0052852, -0.0023290, -0.0053531, -0.0023433, -0.0013834, 0.0012990
8: 0.9854908, 0.9875732, 0.9854430, 0.9875632, -0.0009745, 0.0009150
9: -0.0046071, -0.0027168, -0.0045980, -0.0026734, -0.0008306, 0.0008846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005354, upper bound: 0.0005282
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005354, upper bound: 0.0005282
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033877, 0.0043080, 0.0033409, 0.0042918, -0.0004385, 0.0004469
1: 0.0018117, 0.0019447, 0.0018050, 0.0019423, -0.0000633, 0.0000646
2: 0.0119781, 0.0124869, 0.0119870, 0.0125128, -0.0002471, 0.0002424
3: -0.0022922, -0.0017659, -0.0022829, -0.0017392, -0.0002555, 0.0002507
4: -0.0021253, -0.0015555, -0.0021542, -0.0015656, -0.0002714, 0.0002766
5: 0.0055854, 0.0061246, 0.0055949, 0.0061520, -0.0002618, 0.0002569
6: -0.0001391, 0.0020001, -0.0001013, 0.0021089, -0.0010387, 0.0010191
7: -0.0052807, -0.0023673, -0.0054288, -0.0024187, -0.0013879, 0.0014146
8: 0.9854940, 0.9875464, 0.9853898, 0.9875101, -0.0009777, 0.0009965
9: -0.0045826, -0.0027197, -0.0045498, -0.0026250, -0.0009045, 0.0008875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005256, upper bound: 0.0005279
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005279
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033407, 0.0042954, 0.0033648, 0.0043156, -0.0004221, 0.0003203
1: 0.0018049, 0.0019429, 0.0018084, 0.0019458, -0.0000610, 0.0000463
2: 0.0119851, 0.0125129, 0.0119739, 0.0124995, -0.0001771, 0.0002334
3: -0.0022849, -0.0017390, -0.0022965, -0.0017528, -0.0001831, 0.0002414
4: -0.0021544, -0.0015634, -0.0021394, -0.0015508, -0.0002613, 0.0001983
5: 0.0055928, 0.0061521, 0.0055810, 0.0061380, -0.0001876, 0.0002473
6: -0.0001096, 0.0021094, -0.0001567, 0.0020533, -0.0007444, 0.0009811
7: -0.0054295, -0.0024075, -0.0053531, -0.0023433, -0.0013361, 0.0010138
8: 0.9853891, 0.9875180, 0.9854430, 0.9875632, -0.0009412, 0.0007141
9: -0.0045570, -0.0026246, -0.0045980, -0.0026734, -0.0006482, 0.0008544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005614, upper bound: 0.0005373
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005611, upper bound: 0.0005374
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033421, 0.0042830, 0.0033409, 0.0042918, -0.0004238, 0.0003640
1: 0.0018051, 0.0019411, 0.0018050, 0.0019423, -0.0000612, 0.0000526
2: 0.0119919, 0.0125121, 0.0119870, 0.0125128, -0.0002012, 0.0002343
3: -0.0022779, -0.0017399, -0.0022829, -0.0017392, -0.0002081, 0.0002423
4: -0.0021535, -0.0015710, -0.0021542, -0.0015656, -0.0002624, 0.0002253
5: 0.0056001, 0.0061512, 0.0055949, 0.0061520, -0.0002132, 0.0002483
6: -0.0000808, 0.0021060, -0.0001013, 0.0021089, -0.0008460, 0.0009851
7: -0.0054249, -0.0024466, -0.0054288, -0.0024187, -0.0013416, 0.0011522
8: 0.9853924, 0.9874904, 0.9853898, 0.9875101, -0.0009450, 0.0008117
9: -0.0045319, -0.0026275, -0.0045498, -0.0026250, -0.0007368, 0.0008578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005373
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005374
time: 0.67 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.03 seconds
IS_A1_B1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005460
IS_A1_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005469
IS_A1_B1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005469
IS_A1_B1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005469
IS_A1_B1_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005402
IS_A1_B1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005433
IS_A1_B1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005402
IS_A1_B1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005435, upper bound: 0.0005436
IS_A1_B1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005265
IS_A1_B1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005424, upper bound: 0.0005279
IS_A1_B1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005445, upper bound: 0.0005281
IS_A1_B1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005424, upper bound: 0.0005281
IS_A1_B1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005364, upper bound: 0.0005267
IS_A1_B1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005357, upper bound: 0.0005267
IS_A1_B1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005364, upper bound: 0.0005269
IS_A1_B1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005357, upper bound: 0.0005269
IS_A1_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005460
IS_A1_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005460
IS_A1_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005469
IS_A1_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005469
IS_A1_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005402
IS_A1_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005402
IS_A1_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005436
IS_A1_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005397, upper bound: 0.0005436
IS_A1_B1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005338, upper bound: 0.0005265
IS_A1_B1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005338, upper bound: 0.0005265
IS_A1_B1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005338, upper bound: 0.0005281
IS_A1_B1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005338, upper bound: 0.0005281
IS_A1_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005285, upper bound: 0.0005256
IS_A1_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005285, upper bound: 0.0005256
IS_A1_B1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005285, upper bound: 0.0005269
IS_A1_B1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005285, upper bound: 0.0005269
IS_A1_B2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005620, upper bound: 0.0005590
IS_A1_B2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005620, upper bound: 0.0005590
IS_A1_B2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005622, upper bound: 0.0005546
IS_A1_B2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005622, upper bound: 0.0005554
IS_A1_B2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005592
IS_A1_B2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005593
IS_A1_B2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005554
IS_A1_B2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005605, upper bound: 0.0005565
IS_A1_B2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005556, upper bound: 0.0005506
IS_A1_B2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005556, upper bound: 0.0005507
IS_A1_B2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005486, upper bound: 0.0005497
IS_A1_B2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005486, upper bound: 0.0005499
IS_A1_B2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005635, upper bound: 0.0005674
IS_A1_B2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005635, upper bound: 0.0005675
IS_A1_B2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005590, upper bound: 0.0005674
IS_A1_B2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005590, upper bound: 0.0005675
IS_A1_B2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005436, upper bound: 0.0005444
IS_A1_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005436, upper bound: 0.0005439
IS_A1_B2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005436, upper bound: 0.0005402
IS_A1_B2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005436, upper bound: 0.0005412
IS_A1_B2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005445
IS_A1_B2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005440, upper bound: 0.0005447
IS_A1_B2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005407
IS_A1_B2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005440, upper bound: 0.0005429
IS_A1_B2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005391
IS_A1_B2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005381
IS_A1_B2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005287, upper bound: 0.0005371
IS_A1_B2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005307, upper bound: 0.0005371
IS_A1_B2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005521
IS_A1_B2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005452, upper bound: 0.0005546
IS_A1_B2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005421, upper bound: 0.0005521
IS_A1_B2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005422, upper bound: 0.0005546
IS_A2_B1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005355
IS_A2_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005420, upper bound: 0.0005359
IS_A2_B1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005359
IS_A2_B1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005420, upper bound: 0.0005359
IS_A2_B1_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005286
IS_A2_B1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005306
IS_A2_B1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005287
IS_A2_B1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005307
IS_A2_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005289
IS_A2_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005420, upper bound: 0.0005285
IS_A2_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005438, upper bound: 0.0005285
IS_A2_B1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005420, upper bound: 0.0005285
IS_A2_B1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005202
IS_A2_B1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005207
IS_A2_B1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005411, upper bound: 0.0005204
IS_A2_B1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005416, upper bound: 0.0005210
IS_A2_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005355
IS_A2_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005355
IS_A2_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005359
IS_A2_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005359
IS_A2_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005287
IS_A2_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005287
IS_A2_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005307
IS_A2_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005307
IS_A2_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005289
IS_A2_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005289
IS_A2_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005285
IS_A2_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005285
IS_A2_B1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005204
IS_A2_B1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005204
IS_A2_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005210
IS_A2_B1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005210
IS_A2_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005568
IS_A2_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005568
IS_A2_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005485
IS_A2_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005485
IS_A2_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005624
IS_A2_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005624
IS_A2_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005575
IS_A2_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005575
IS_A2_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005437
IS_A2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005437
IS_A2_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005437
IS_A2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005437
IS_A2_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005551
IS_A2_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005551
IS_A2_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005551
IS_A2_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005551
IS_A2_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005265, upper bound: 0.0005378
IS_A2_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005379
IS_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005256, upper bound: 0.0005325
IS_A2_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005325
IS_A2_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005437
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005438
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005413
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005414
IS_A2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005354, upper bound: 0.0005282
IS_A2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005354, upper bound: 0.0005282
IS_A2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005256, upper bound: 0.0005279
IS_A2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005279
IS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005614, upper bound: 0.0005373
IS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005611, upper bound: 0.0005374
IS_A2_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005373
IS_A2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005374

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033463, 0.0042226, 0.0033782, 0.0042741, -0.0003502, 0.0003104
1: 0.0018057, 0.0019323, 0.0018103, 0.0019398, -0.0000506, 0.0000448
2: 0.0120253, 0.0125098, 0.0119968, 0.0124922, -0.0001716, 0.0001936
3: -0.0022433, -0.0017422, -0.0022728, -0.0017605, -0.0001775, 0.0002003
4: -0.0021509, -0.0016084, -0.0021311, -0.0015765, -0.0002168, 0.0001921
5: 0.0056355, 0.0061488, 0.0056053, 0.0061301, -0.0001818, 0.0002052
6: 0.0000595, 0.0020964, -0.0000602, 0.0020222, -0.0007214, 0.0008141
7: -0.0054118, -0.0026377, -0.0053108, -0.0024747, -0.0011087, 0.0009825
8: 0.9854017, 0.9873558, 0.9854728, 0.9874706, -0.0007810, 0.0006921
9: -0.0044097, -0.0026359, -0.0045140, -0.0027005, -0.0006282, 0.0007089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005531
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005530
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033442, 0.0042196, 0.0033781, 0.0042736, -0.0003637, 0.0003147
1: 0.0018054, 0.0019319, 0.0018103, 0.0019397, -0.0000525, 0.0000455
2: 0.0120270, 0.0125109, 0.0119971, 0.0124922, -0.0001740, 0.0002011
3: -0.0022416, -0.0017411, -0.0022725, -0.0017604, -0.0001800, 0.0002080
4: -0.0021521, -0.0016103, -0.0021312, -0.0015768, -0.0002251, 0.0001948
5: 0.0056372, 0.0061500, 0.0056056, 0.0061302, -0.0001844, 0.0002130
6: 0.0000665, 0.0021011, -0.0000591, 0.0020224, -0.0007315, 0.0008453
7: -0.0054182, -0.0026473, -0.0053111, -0.0024763, -0.0011512, 0.0009963
8: 0.9853972, 0.9873491, 0.9854727, 0.9874696, -0.0008110, 0.0007018
9: -0.0044036, -0.0026318, -0.0045130, -0.0027003, -0.0006370, 0.0007361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005538
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005538
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033227, 0.0042002, 0.0033787, 0.0042663, -0.0004138, 0.0003295
1: 0.0018023, 0.0019291, 0.0018104, 0.0019387, -0.0000598, 0.0000476
2: 0.0120377, 0.0125228, 0.0120011, 0.0124919, -0.0001822, 0.0002288
3: -0.0022305, -0.0017288, -0.0022683, -0.0017608, -0.0001884, 0.0002366
4: -0.0021655, -0.0016223, -0.0021308, -0.0015814, -0.0002561, 0.0002040
5: 0.0056486, 0.0061626, 0.0056099, 0.0061298, -0.0001930, 0.0002424
6: 0.0001117, 0.0021510, -0.0000421, 0.0020209, -0.0007659, 0.0009617
7: -0.0054863, -0.0027088, -0.0053091, -0.0024994, -0.0013098, 0.0010431
8: 0.9853493, 0.9873057, 0.9854741, 0.9874532, -0.0009227, 0.0007348
9: -0.0043643, -0.0025883, -0.0044981, -0.0027016, -0.0006670, 0.0008375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005417, upper bound: 0.0005469
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005417, upper bound: 0.0005469
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033227, 0.0041990, 0.0033758, 0.0042657, -0.0004170, 0.0003400
1: 0.0018023, 0.0019289, 0.0018100, 0.0019386, -0.0000602, 0.0000491
2: 0.0120383, 0.0125228, 0.0120014, 0.0124935, -0.0001880, 0.0002306
3: -0.0022299, -0.0017288, -0.0022680, -0.0017591, -0.0001944, 0.0002385
4: -0.0021655, -0.0016230, -0.0021326, -0.0015817, -0.0002581, 0.0002105
5: 0.0056493, 0.0061626, 0.0056102, 0.0061315, -0.0001992, 0.0002443
6: 0.0001143, 0.0021511, -0.0000408, 0.0020277, -0.0007902, 0.0009693
7: -0.0054864, -0.0027123, -0.0053182, -0.0025012, -0.0013201, 0.0010762
8: 0.9853492, 0.9873032, 0.9854677, 0.9874520, -0.0009299, 0.0007581
9: -0.0043620, -0.0025882, -0.0044970, -0.0026957, -0.0006882, 0.0008441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005469
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005469
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033253, 0.0041951, 0.0033795, 0.0042597, -0.0003921, 0.0003136
1: 0.0018027, 0.0019284, 0.0018105, 0.0019377, -0.0000567, 0.0000453
2: 0.0120405, 0.0125214, 0.0120048, 0.0124914, -0.0001734, 0.0002168
3: -0.0022276, -0.0017303, -0.0022645, -0.0017612, -0.0001793, 0.0002242
4: -0.0021638, -0.0016255, -0.0021303, -0.0015855, -0.0002427, 0.0001941
5: 0.0056516, 0.0061611, 0.0056137, 0.0061293, -0.0001837, 0.0002297
6: 0.0001235, 0.0021450, -0.0000267, 0.0020191, -0.0007289, 0.0009114
7: -0.0054780, -0.0027249, -0.0053065, -0.0025204, -0.0012413, 0.0009927
8: 0.9853550, 0.9872944, 0.9854758, 0.9874385, -0.0008744, 0.0006993
9: -0.0043540, -0.0025935, -0.0044848, -0.0027032, -0.0006348, 0.0007937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005474
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005474
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033220, 0.0041930, 0.0033794, 0.0042592, -0.0004034, 0.0003186
1: 0.0018022, 0.0019281, 0.0018105, 0.0019376, -0.0000583, 0.0000460
2: 0.0120417, 0.0125232, 0.0120050, 0.0124915, -0.0001761, 0.0002230
3: -0.0022264, -0.0017283, -0.0022643, -0.0017612, -0.0001822, 0.0002307
4: -0.0021659, -0.0016268, -0.0021304, -0.0015857, -0.0002497, 0.0001972
5: 0.0056528, 0.0061630, 0.0056140, 0.0061294, -0.0001866, 0.0002363
6: 0.0001284, 0.0021529, -0.0000256, 0.0020193, -0.0007405, 0.0009376
7: -0.0054887, -0.0027316, -0.0053068, -0.0025218, -0.0012769, 0.0010085
8: 0.9853475, 0.9872896, 0.9854756, 0.9874374, -0.0008995, 0.0007104
9: -0.0043497, -0.0025867, -0.0044838, -0.0027030, -0.0006449, 0.0008165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005498
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005498
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033005, 0.0041667, 0.0033795, 0.0042597, -0.0004521, 0.0003243
1: 0.0017991, 0.0019243, 0.0018105, 0.0019377, -0.0000653, 0.0000469
2: 0.0120562, 0.0125351, 0.0120048, 0.0124914, -0.0001793, 0.0002499
3: -0.0022114, -0.0017161, -0.0022645, -0.0017612, -0.0001855, 0.0002585
4: -0.0021792, -0.0016430, -0.0021303, -0.0015855, -0.0002799, 0.0002008
5: 0.0056682, 0.0061756, 0.0056137, 0.0061293, -0.0001900, 0.0002648
6: 0.0001894, 0.0022027, -0.0000267, 0.0020191, -0.0007539, 0.0010508
7: -0.0055566, -0.0028146, -0.0053065, -0.0025204, -0.0014311, 0.0010267
8: 0.9852997, 0.9872312, 0.9854758, 0.9874385, -0.0010081, 0.0007232
9: -0.0042966, -0.0025433, -0.0044848, -0.0027032, -0.0006565, 0.0009151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005402
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005402
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0032956, 0.0041657, 0.0033794, 0.0042592, -0.0004629, 0.0003293
1: 0.0017984, 0.0019241, 0.0018105, 0.0019376, -0.0000669, 0.0000476
2: 0.0120568, 0.0125378, 0.0120050, 0.0124915, -0.0001820, 0.0002559
3: -0.0022108, -0.0017133, -0.0022643, -0.0017612, -0.0001883, 0.0002647
4: -0.0021823, -0.0016436, -0.0021304, -0.0015857, -0.0002865, 0.0002038
5: 0.0056688, 0.0061785, 0.0056140, 0.0061294, -0.0001929, 0.0002711
6: 0.0001918, 0.0022142, -0.0000256, 0.0020193, -0.0007653, 0.0010758
7: -0.0055722, -0.0028179, -0.0053068, -0.0025218, -0.0014652, 0.0010422
8: 0.9852887, 0.9872289, 0.9854756, 0.9874374, -0.0010321, 0.0007342
9: -0.0042945, -0.0025333, -0.0044838, -0.0027030, -0.0006664, 0.0009369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005436
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005436
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033449, 0.0042306, 0.0034089, 0.0043391, -0.0005240, 0.0003870
1: 0.0018055, 0.0019335, 0.0018148, 0.0019492, -0.0000757, 0.0000559
2: 0.0120209, 0.0125105, 0.0119609, 0.0124752, -0.0002140, 0.0002897
3: -0.0022479, -0.0017415, -0.0023099, -0.0017780, -0.0002213, 0.0002996
4: -0.0021517, -0.0016034, -0.0021121, -0.0015363, -0.0003243, 0.0002396
5: 0.0056307, 0.0061496, 0.0055672, 0.0061122, -0.0002267, 0.0003069
6: 0.0000408, 0.0020995, -0.0002112, 0.0019509, -0.0008995, 0.0012178
7: -0.0054161, -0.0026123, -0.0052137, -0.0022690, -0.0016586, 0.0012250
8: 0.9853987, 0.9873737, 0.9855413, 0.9876155, -0.0011683, 0.0008629
9: -0.0044260, -0.0026332, -0.0046455, -0.0027626, -0.0007833, 0.0010606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005443, upper bound: 0.0005475
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005443, upper bound: 0.0005475
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033428, 0.0042278, 0.0034088, 0.0043385, -0.0005289, 0.0003914
1: 0.0018052, 0.0019331, 0.0018148, 0.0019491, -0.0000764, 0.0000565
2: 0.0120224, 0.0125117, 0.0119612, 0.0124752, -0.0002164, 0.0002924
3: -0.0022463, -0.0017403, -0.0023096, -0.0017780, -0.0002238, 0.0003024
4: -0.0021530, -0.0016052, -0.0021122, -0.0015367, -0.0003274, 0.0002423
5: 0.0056324, 0.0061508, 0.0055676, 0.0061122, -0.0002293, 0.0003098
6: 0.0000475, 0.0021043, -0.0002099, 0.0019511, -0.0009097, 0.0012293
7: -0.0054226, -0.0026214, -0.0052139, -0.0022709, -0.0016741, 0.0012389
8: 0.9853941, 0.9873673, 0.9855411, 0.9876142, -0.0011793, 0.0008727
9: -0.0044202, -0.0026290, -0.0046443, -0.0027625, -0.0007922, 0.0010705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005444, upper bound: 0.0005478
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005444, upper bound: 0.0005478
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033214, 0.0042086, 0.0034094, 0.0043315, -0.0005903, 0.0004046
1: 0.0018022, 0.0019303, 0.0018149, 0.0019481, -0.0000853, 0.0000585
2: 0.0120331, 0.0125235, 0.0119651, 0.0124749, -0.0002237, 0.0003264
3: -0.0022353, -0.0017280, -0.0023056, -0.0017784, -0.0002314, 0.0003375
4: -0.0021662, -0.0016171, -0.0021118, -0.0015410, -0.0003654, 0.0002505
5: 0.0056437, 0.0061634, 0.0055716, 0.0061118, -0.0002370, 0.0003458
6: 0.0000922, 0.0021540, -0.0001937, 0.0019495, -0.0009405, 0.0013720
7: -0.0054903, -0.0026822, -0.0052118, -0.0022929, -0.0018686, 0.0012808
8: 0.9853464, 0.9873244, 0.9855426, 0.9875987, -0.0013163, 0.0009022
9: -0.0043813, -0.0025857, -0.0046302, -0.0027638, -0.0008190, 0.0011948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005409, upper bound: 0.0005281
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005409, upper bound: 0.0005281
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033214, 0.0042075, 0.0034062, 0.0043307, -0.0005933, 0.0004048
1: 0.0018021, 0.0019302, 0.0018144, 0.0019480, -0.0000857, 0.0000585
2: 0.0120336, 0.0125235, 0.0119655, 0.0124767, -0.0002238, 0.0003280
3: -0.0022347, -0.0017280, -0.0023052, -0.0017765, -0.0002314, 0.0003393
4: -0.0021663, -0.0016178, -0.0021138, -0.0015415, -0.0003673, 0.0002506
5: 0.0056443, 0.0061634, 0.0055721, 0.0061137, -0.0002371, 0.0003476
6: 0.0000946, 0.0021541, -0.0001918, 0.0019572, -0.0009408, 0.0013791
7: -0.0054905, -0.0026856, -0.0052222, -0.0022955, -0.0018782, 0.0012812
8: 0.9853463, 0.9873221, 0.9855353, 0.9875969, -0.0013230, 0.0009025
9: -0.0043791, -0.0025856, -0.0046286, -0.0027571, -0.0008193, 0.0012010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005403, upper bound: 0.0005281
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005403, upper bound: 0.0005281
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033458, 0.0042238, 0.0033857, 0.0043072, -0.0005178, 0.0004220
1: 0.0018057, 0.0019325, 0.0018114, 0.0019446, -0.0000748, 0.0000610
2: 0.0120246, 0.0125101, 0.0119785, 0.0124880, -0.0002333, 0.0002863
3: -0.0022440, -0.0017420, -0.0022917, -0.0017648, -0.0002413, 0.0002961
4: -0.0021512, -0.0016077, -0.0021265, -0.0015561, -0.0003205, 0.0002612
5: 0.0056348, 0.0061491, 0.0055859, 0.0061257, -0.0002472, 0.0003033
6: 0.0000567, 0.0020975, -0.0001371, 0.0020048, -0.0009808, 0.0012034
7: -0.0054133, -0.0026339, -0.0052870, -0.0023700, -0.0016390, 0.0013358
8: 0.9854006, 0.9873585, 0.9854896, 0.9875444, -0.0011545, 0.0009410
9: -0.0044121, -0.0026349, -0.0045809, -0.0027157, -0.0008541, 0.0010480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005478
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005478
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033457, 0.0042224, 0.0033804, 0.0043067, -0.0005216, 0.0004256
1: 0.0018057, 0.0019323, 0.0018107, 0.0019445, -0.0000754, 0.0000615
2: 0.0120254, 0.0125101, 0.0119788, 0.0124909, -0.0002353, 0.0002884
3: -0.0022432, -0.0017419, -0.0022914, -0.0017618, -0.0002434, 0.0002982
4: -0.0021512, -0.0016086, -0.0021297, -0.0015564, -0.0003229, 0.0002635
5: 0.0056356, 0.0061491, 0.0055862, 0.0061288, -0.0002493, 0.0003055
6: 0.0000601, 0.0020976, -0.0001360, 0.0020169, -0.0009893, 0.0012123
7: -0.0054135, -0.0026385, -0.0053036, -0.0023715, -0.0016511, 0.0013473
8: 0.9854004, 0.9873553, 0.9854779, 0.9875433, -0.0011630, 0.0009491
9: -0.0044092, -0.0026348, -0.0045799, -0.0027051, -0.0008615, 0.0010557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005400, upper bound: 0.0005478
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005400, upper bound: 0.0005478
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0041949, 0.0033857, 0.0043072, -0.0005913, 0.0004277
1: 0.0018024, 0.0019283, 0.0018114, 0.0019446, -0.0000854, 0.0000618
2: 0.0120406, 0.0125228, 0.0119785, 0.0124880, -0.0002364, 0.0003269
3: -0.0022275, -0.0017288, -0.0022917, -0.0017648, -0.0002445, 0.0003381
4: -0.0021654, -0.0016256, -0.0021265, -0.0015561, -0.0003660, 0.0002647
5: 0.0056517, 0.0061625, 0.0055859, 0.0061257, -0.0002505, 0.0003464
6: 0.0001239, 0.0021508, -0.0001371, 0.0020048, -0.0009940, 0.0013743
7: -0.0054859, -0.0027255, -0.0052870, -0.0023700, -0.0018716, 0.0013538
8: 0.9853495, 0.9872940, 0.9854896, 0.9875444, -0.0013184, 0.0009536
9: -0.0043536, -0.0025885, -0.0045809, -0.0027157, -0.0008656, 0.0011968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0041939, 0.0033804, 0.0043067, -0.0005952, 0.0004311
1: 0.0018023, 0.0019282, 0.0018107, 0.0019445, -0.0000860, 0.0000623
2: 0.0120412, 0.0125228, 0.0119788, 0.0124909, -0.0002383, 0.0003291
3: -0.0022269, -0.0017288, -0.0022914, -0.0017618, -0.0002465, 0.0003404
4: -0.0021654, -0.0016262, -0.0021297, -0.0015564, -0.0003685, 0.0002668
5: 0.0056523, 0.0061626, 0.0055862, 0.0061288, -0.0002525, 0.0003487
6: 0.0001262, 0.0021509, -0.0001360, 0.0020169, -0.0010020, 0.0013835
7: -0.0054861, -0.0027286, -0.0053036, -0.0023715, -0.0018842, 0.0013646
8: 0.9853494, 0.9872918, 0.9854779, 0.9875433, -0.0013273, 0.0009612
9: -0.0043516, -0.0025884, -0.0045799, -0.0027051, -0.0008725, 0.0012048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042543, 0.0033572, 0.0042401, -0.0002915, 0.0003892
1: 0.0018105, 0.0019369, 0.0018073, 0.0019349, -0.0000421, 0.0000562
2: 0.0120078, 0.0124915, 0.0120156, 0.0125038, -0.0002152, 0.0001612
3: -0.0022615, -0.0017612, -0.0022533, -0.0017485, -0.0002226, 0.0001667
4: -0.0021304, -0.0015888, -0.0021441, -0.0015976, -0.0001805, 0.0002410
5: 0.0056169, 0.0061294, 0.0056252, 0.0061424, -0.0002280, 0.0001708
6: -0.0000142, 0.0020194, 0.0000190, 0.0020710, -0.0009047, 0.0006776
7: -0.0053070, -0.0025373, -0.0053772, -0.0025825, -0.0009228, 0.0012321
8: 0.9854755, 0.9874265, 0.9854261, 0.9873947, -0.0006501, 0.0008680
9: -0.0044739, -0.0027029, -0.0044450, -0.0026580, -0.0007879, 0.0005901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005460
time: 0.68 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005460
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033438, 0.0042216, 0.0033572, 0.0042401, -0.0003886, 0.0004131
1: 0.0018054, 0.0019322, 0.0018073, 0.0019349, -0.0000561, 0.0000597
2: 0.0120259, 0.0125112, 0.0120156, 0.0125038, -0.0002284, 0.0002149
3: -0.0022427, -0.0017408, -0.0022533, -0.0017485, -0.0002362, 0.0002222
4: -0.0021524, -0.0016091, -0.0021441, -0.0015976, -0.0002406, 0.0002557
5: 0.0056361, 0.0061503, 0.0056252, 0.0061424, -0.0002420, 0.0002276
6: 0.0000619, 0.0021021, 0.0000190, 0.0020710, -0.0009603, 0.0009032
7: -0.0054196, -0.0026411, -0.0053772, -0.0025825, -0.0012301, 0.0013078
8: 0.9853963, 0.9873534, 0.9854261, 0.9873947, -0.0008665, 0.0009212
9: -0.0044076, -0.0026309, -0.0044450, -0.0026580, -0.0008362, 0.0007866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005460
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005460
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033766, 0.0042538, 0.0033571, 0.0042396, -0.0003115, 0.0003923
1: 0.0018101, 0.0019369, 0.0018073, 0.0019348, -0.0000450, 0.0000567
2: 0.0120080, 0.0124930, 0.0120159, 0.0125038, -0.0002169, 0.0001722
3: -0.0022612, -0.0017596, -0.0022530, -0.0017484, -0.0002243, 0.0001781
4: -0.0021321, -0.0015891, -0.0021442, -0.0015979, -0.0001928, 0.0002429
5: 0.0056172, 0.0061310, 0.0056255, 0.0061425, -0.0002298, 0.0001825
6: -0.0000130, 0.0020258, 0.0000201, 0.0020711, -0.0009119, 0.0007240
7: -0.0053156, -0.0025390, -0.0053774, -0.0025841, -0.0009860, 0.0012419
8: 0.9854693, 0.9874254, 0.9854259, 0.9873935, -0.0006945, 0.0008749
9: -0.0044729, -0.0026974, -0.0044440, -0.0026579, -0.0007941, 0.0006305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005469
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005469
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033429, 0.0042192, 0.0033571, 0.0042396, -0.0003994, 0.0004174
1: 0.0018053, 0.0019318, 0.0018073, 0.0019348, -0.0000577, 0.0000603
2: 0.0120272, 0.0125117, 0.0120159, 0.0125038, -0.0002308, 0.0002208
3: -0.0022414, -0.0017403, -0.0022530, -0.0017484, -0.0002387, 0.0002284
4: -0.0021530, -0.0016105, -0.0021442, -0.0015979, -0.0002473, 0.0002584
5: 0.0056375, 0.0061508, 0.0056255, 0.0061425, -0.0002445, 0.0002340
6: 0.0000675, 0.0021042, 0.0000201, 0.0020711, -0.0009701, 0.0009284
7: -0.0054225, -0.0026486, -0.0053774, -0.0025841, -0.0012644, 0.0013212
8: 0.9853942, 0.9873481, 0.9854259, 0.9873935, -0.0008906, 0.0009307
9: -0.0044028, -0.0026291, -0.0044440, -0.0026579, -0.0008448, 0.0008085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005469
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005469
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033585, 0.0042268, 0.0033586, 0.0042264, -0.0003380, 0.0003929
1: 0.0018075, 0.0019330, 0.0018075, 0.0019329, -0.0000488, 0.0000568
2: 0.0120229, 0.0125030, 0.0120232, 0.0125030, -0.0002172, 0.0001869
3: -0.0022458, -0.0017492, -0.0022455, -0.0017493, -0.0002247, 0.0001933
4: -0.0021433, -0.0016058, -0.0021433, -0.0016061, -0.0002092, 0.0002432
5: 0.0056330, 0.0061416, 0.0056332, 0.0061416, -0.0002302, 0.0001980
6: 0.0000496, 0.0020679, 0.0000507, 0.0020678, -0.0009133, 0.0007857
7: -0.0053730, -0.0026243, -0.0053729, -0.0026258, -0.0010700, 0.0012438
8: 0.9854290, 0.9873653, 0.9854292, 0.9873642, -0.0007537, 0.0008762
9: -0.0044183, -0.0026607, -0.0044173, -0.0026608, -0.0007953, 0.0006842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005402
time: 0.71 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005402
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033233, 0.0041944, 0.0033586, 0.0042264, -0.0004254, 0.0004144
1: 0.0018024, 0.0019283, 0.0018075, 0.0019329, -0.0000615, 0.0000599
2: 0.0120409, 0.0125225, 0.0120232, 0.0125030, -0.0002291, 0.0002352
3: -0.0022272, -0.0017291, -0.0022455, -0.0017493, -0.0002370, 0.0002432
4: -0.0021651, -0.0016259, -0.0021433, -0.0016061, -0.0002633, 0.0002565
5: 0.0056520, 0.0061623, 0.0056332, 0.0061416, -0.0002428, 0.0002492
6: 0.0001251, 0.0021498, 0.0000507, 0.0020678, -0.0009632, 0.0009887
7: -0.0054845, -0.0027271, -0.0053729, -0.0026258, -0.0013466, 0.0013118
8: 0.9853505, 0.9872928, 0.9854292, 0.9873642, -0.0009486, 0.0009240
9: -0.0043525, -0.0025894, -0.0044173, -0.0026608, -0.0008388, 0.0008610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005402
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005402
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033538, 0.0042281, 0.0033585, 0.0042264, -0.0003580, 0.0003971
1: 0.0018068, 0.0019331, 0.0018075, 0.0019329, -0.0000517, 0.0000574
2: 0.0120223, 0.0125056, 0.0120232, 0.0125030, -0.0002195, 0.0001979
3: -0.0022465, -0.0017466, -0.0022455, -0.0017492, -0.0002271, 0.0002047
4: -0.0021462, -0.0016050, -0.0021433, -0.0016061, -0.0002216, 0.0002458
5: 0.0056322, 0.0061444, 0.0056332, 0.0061417, -0.0002326, 0.0002097
6: 0.0000468, 0.0020788, 0.0000507, 0.0020680, -0.0009229, 0.0008320
7: -0.0053879, -0.0026204, -0.0053731, -0.0026258, -0.0011331, 0.0012569
8: 0.9854186, 0.9873680, 0.9854289, 0.9873642, -0.0007982, 0.0008854
9: -0.0044208, -0.0026512, -0.0044174, -0.0026606, -0.0008037, 0.0007246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005433
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005436
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033213, 0.0041926, 0.0033585, 0.0042264, -0.0004348, 0.0004198
1: 0.0018021, 0.0019280, 0.0018075, 0.0019329, -0.0000628, 0.0000606
2: 0.0120419, 0.0125236, 0.0120232, 0.0125030, -0.0002321, 0.0002404
3: -0.0022262, -0.0017280, -0.0022455, -0.0017492, -0.0002400, 0.0002486
4: -0.0021663, -0.0016270, -0.0021433, -0.0016061, -0.0002692, 0.0002599
5: 0.0056530, 0.0061634, 0.0056332, 0.0061417, -0.0002459, 0.0002547
6: 0.0001291, 0.0021543, 0.0000507, 0.0020680, -0.0009757, 0.0010106
7: -0.0054907, -0.0027326, -0.0053731, -0.0026258, -0.0013764, 0.0013288
8: 0.9853461, 0.9872890, 0.9854289, 0.9873642, -0.0009696, 0.0009360
9: -0.0043491, -0.0025854, -0.0044174, -0.0026606, -0.0008497, 0.0008801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005433
time: 0.67 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005436
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033780, 0.0042621, 0.0033881, 0.0043066, -0.0004646, 0.0004264
1: 0.0018103, 0.0019381, 0.0018118, 0.0019445, -0.0000671, 0.0000616
2: 0.0120034, 0.0124923, 0.0119789, 0.0124867, -0.0002358, 0.0002569
3: -0.0022659, -0.0017604, -0.0022914, -0.0017662, -0.0002438, 0.0002657
4: -0.0021312, -0.0015839, -0.0021250, -0.0015564, -0.0002876, 0.0002640
5: 0.0056123, 0.0061302, 0.0055863, 0.0061243, -0.0002498, 0.0002722
6: -0.0000324, 0.0020226, -0.0001357, 0.0019991, -0.0009911, 0.0010799
7: -0.0053113, -0.0025126, -0.0052793, -0.0023719, -0.0014707, 0.0013499
8: 0.9854725, 0.9874439, 0.9854950, 0.9875431, -0.0010360, 0.0009509
9: -0.0044897, -0.0027001, -0.0045797, -0.0027206, -0.0008631, 0.0009404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005265
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005265
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033424, 0.0042296, 0.0033881, 0.0043066, -0.0005615, 0.0004500
1: 0.0018052, 0.0019334, 0.0018118, 0.0019445, -0.0000811, 0.0000650
2: 0.0120214, 0.0125119, 0.0119789, 0.0124867, -0.0002488, 0.0003105
3: -0.0022473, -0.0017400, -0.0022914, -0.0017662, -0.0002573, 0.0003211
4: -0.0021533, -0.0016041, -0.0021250, -0.0015564, -0.0003476, 0.0002786
5: 0.0056313, 0.0061511, 0.0055863, 0.0061243, -0.0002636, 0.0003290
6: 0.0000432, 0.0021053, -0.0001357, 0.0019991, -0.0010460, 0.0013052
7: -0.0054240, -0.0026155, -0.0052793, -0.0023719, -0.0017775, 0.0014246
8: 0.9853930, 0.9873714, 0.9854950, 0.9875431, -0.0012521, 0.0010035
9: -0.0044239, -0.0026281, -0.0045797, -0.0027206, -0.0009109, 0.0011366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005265
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005265
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033752, 0.0042619, 0.0033880, 0.0043058, -0.0004756, 0.0004289
1: 0.0018099, 0.0019380, 0.0018118, 0.0019444, -0.0000687, 0.0000620
2: 0.0120036, 0.0124938, 0.0119793, 0.0124867, -0.0002371, 0.0002630
3: -0.0022658, -0.0017588, -0.0022909, -0.0017661, -0.0002453, 0.0002720
4: -0.0021330, -0.0015841, -0.0021250, -0.0015569, -0.0002944, 0.0002655
5: 0.0056124, 0.0061319, 0.0055867, 0.0061243, -0.0002513, 0.0002786
6: -0.0000319, 0.0020291, -0.0001339, 0.0019993, -0.0009970, 0.0011055
7: -0.0053202, -0.0025133, -0.0052796, -0.0023743, -0.0015056, 0.0013578
8: 0.9854662, 0.9874434, 0.9854948, 0.9875413, -0.0010606, 0.0009565
9: -0.0044893, -0.0026945, -0.0045781, -0.0027204, -0.0008682, 0.0009627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005278
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005281
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033414, 0.0042274, 0.0033880, 0.0043058, -0.0005636, 0.0004541
1: 0.0018050, 0.0019330, 0.0018118, 0.0019444, -0.0000814, 0.0000656
2: 0.0120227, 0.0125125, 0.0119793, 0.0124867, -0.0002511, 0.0003116
3: -0.0022461, -0.0017395, -0.0022909, -0.0017661, -0.0002597, 0.0003223
4: -0.0021539, -0.0016055, -0.0021250, -0.0015569, -0.0003489, 0.0002811
5: 0.0056327, 0.0061516, 0.0055867, 0.0061243, -0.0002660, 0.0003302
6: 0.0000485, 0.0021076, -0.0001339, 0.0019993, -0.0010554, 0.0013099
7: -0.0054270, -0.0026227, -0.0052796, -0.0023743, -0.0017840, 0.0014374
8: 0.9853910, 0.9873664, 0.9854948, 0.9875413, -0.0012567, 0.0010125
9: -0.0044193, -0.0026262, -0.0045781, -0.0027204, -0.0009191, 0.0011408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005278
time: 0.70 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005281
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042476, 0.0033625, 0.0042838, -0.0004665, 0.0004552
1: 0.0018105, 0.0019360, 0.0018081, 0.0019412, -0.0000674, 0.0000658
2: 0.0120115, 0.0124915, 0.0119914, 0.0125008, -0.0002517, 0.0002579
3: -0.0022576, -0.0017612, -0.0022783, -0.0017515, -0.0002603, 0.0002667
4: -0.0021304, -0.0015929, -0.0021408, -0.0015705, -0.0002888, 0.0002818
5: 0.0056208, 0.0061294, 0.0055996, 0.0061393, -0.0002667, 0.0002733
6: 0.0000014, 0.0020194, -0.0000828, 0.0020587, -0.0010580, 0.0010842
7: -0.0053070, -0.0025586, -0.0053604, -0.0024440, -0.0014766, 0.0014409
8: 0.9854755, 0.9874116, 0.9854378, 0.9874923, -0.0010402, 0.0010150
9: -0.0044603, -0.0027029, -0.0045336, -0.0026687, -0.0009214, 0.0009442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005256
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005256
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033439, 0.0042156, 0.0033625, 0.0042838, -0.0005631, 0.0004759
1: 0.0018054, 0.0019313, 0.0018081, 0.0019412, -0.0000814, 0.0000688
2: 0.0120292, 0.0125111, 0.0119914, 0.0125008, -0.0002631, 0.0003113
3: -0.0022393, -0.0017409, -0.0022783, -0.0017515, -0.0002721, 0.0003220
4: -0.0021524, -0.0016128, -0.0021408, -0.0015705, -0.0003486, 0.0002946
5: 0.0056396, 0.0061502, 0.0055996, 0.0061393, -0.0002788, 0.0003299
6: 0.0000758, 0.0021020, -0.0000828, 0.0020587, -0.0011061, 0.0013088
7: -0.0054194, -0.0026600, -0.0053604, -0.0024440, -0.0017825, 0.0015065
8: 0.9853964, 0.9873401, 0.9854378, 0.9874923, -0.0012556, 0.0010612
9: -0.0043955, -0.0026310, -0.0045336, -0.0026687, -0.0009633, 0.0011398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005256
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005256
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033766, 0.0042485, 0.0033623, 0.0042835, -0.0004768, 0.0004574
1: 0.0018101, 0.0019361, 0.0018081, 0.0019411, -0.0000689, 0.0000661
2: 0.0120110, 0.0124930, 0.0119916, 0.0125009, -0.0002529, 0.0002636
3: -0.0022582, -0.0017596, -0.0022781, -0.0017514, -0.0002615, 0.0002727
4: -0.0021321, -0.0015924, -0.0021410, -0.0015707, -0.0002952, 0.0002831
5: 0.0056203, 0.0061310, 0.0055998, 0.0061394, -0.0002679, 0.0002793
6: -0.0000008, 0.0020258, -0.0000820, 0.0020591, -0.0010631, 0.0011083
7: -0.0053157, -0.0025556, -0.0053610, -0.0024451, -0.0015094, 0.0014478
8: 0.9854693, 0.9874136, 0.9854374, 0.9874915, -0.0010633, 0.0010199
9: -0.0044622, -0.0026973, -0.0045329, -0.0026684, -0.0009258, 0.0009651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005267
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005269
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033429, 0.0042133, 0.0033623, 0.0042835, -0.0005646, 0.0004806
1: 0.0018053, 0.0019310, 0.0018081, 0.0019411, -0.0000816, 0.0000694
2: 0.0120304, 0.0125117, 0.0119916, 0.0125009, -0.0002657, 0.0003121
3: -0.0022380, -0.0017403, -0.0022781, -0.0017514, -0.0002748, 0.0003228
4: -0.0021530, -0.0016142, -0.0021410, -0.0015707, -0.0003495, 0.0002975
5: 0.0056409, 0.0061508, 0.0055998, 0.0061394, -0.0002815, 0.0003307
6: 0.0000811, 0.0021042, -0.0000820, 0.0020591, -0.0011170, 0.0013122
7: -0.0054224, -0.0026672, -0.0053610, -0.0024451, -0.0017871, 0.0015212
8: 0.9853942, 0.9873350, 0.9854374, 0.9874915, -0.0012589, 0.0010716
9: -0.0043909, -0.0026291, -0.0045329, -0.0026684, -0.0009727, 0.0011427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005267
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005269
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033791, 0.0042717, 0.0033423, 0.0042429, -0.0003369, 0.0003950
1: 0.0018105, 0.0019394, 0.0018052, 0.0019353, -0.0000487, 0.0000571
2: 0.0119981, 0.0124916, 0.0120141, 0.0125120, -0.0002184, 0.0001863
3: -0.0022714, -0.0017610, -0.0022549, -0.0017399, -0.0002259, 0.0001926
4: -0.0021305, -0.0015780, -0.0021534, -0.0015958, -0.0002085, 0.0002445
5: 0.0056067, 0.0061296, 0.0056236, 0.0061512, -0.0002314, 0.0001974
6: -0.0000547, 0.0020200, 0.0000123, 0.0021057, -0.0009181, 0.0007831
7: -0.0053078, -0.0024822, -0.0054245, -0.0025735, -0.0010665, 0.0012504
8: 0.9854749, 0.9874653, 0.9853927, 0.9874010, -0.0007512, 0.0008808
9: -0.0045091, -0.0027024, -0.0044508, -0.0026278, -0.0007995, 0.0006819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005453, upper bound: 0.0005437
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005437
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033581, 0.0042372, 0.0033423, 0.0042429, -0.0004174, 0.0004239
1: 0.0018075, 0.0019345, 0.0018052, 0.0019353, -0.0000603, 0.0000612
2: 0.0120172, 0.0125032, 0.0120141, 0.0125120, -0.0002344, 0.0002308
3: -0.0022517, -0.0017490, -0.0022549, -0.0017399, -0.0002424, 0.0002387
4: -0.0021435, -0.0015994, -0.0021534, -0.0015958, -0.0002584, 0.0002624
5: 0.0056269, 0.0061419, 0.0056236, 0.0061512, -0.0002483, 0.0002445
6: 0.0000256, 0.0020687, 0.0000123, 0.0021057, -0.0009852, 0.0009702
7: -0.0053742, -0.0025915, -0.0054245, -0.0025735, -0.0013214, 0.0013418
8: 0.9854282, 0.9873884, 0.9853927, 0.9874010, -0.0009308, 0.0009452
9: -0.0044393, -0.0026600, -0.0044508, -0.0026278, -0.0008580, 0.0008449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005453, upper bound: 0.0005438
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005439
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033579, 0.0042438, 0.0033437, 0.0042289, -0.0003773, 0.0003986
1: 0.0018074, 0.0019354, 0.0018054, 0.0019332, -0.0000545, 0.0000576
2: 0.0120136, 0.0125033, 0.0120218, 0.0125112, -0.0002204, 0.0002086
3: -0.0022555, -0.0017489, -0.0022469, -0.0017408, -0.0002279, 0.0002158
4: -0.0021437, -0.0015953, -0.0021525, -0.0016046, -0.0002336, 0.0002467
5: 0.0056230, 0.0061420, 0.0056318, 0.0061503, -0.0002335, 0.0002210
6: 0.0000102, 0.0020692, 0.0000450, 0.0021024, -0.0009264, 0.0008770
7: -0.0053748, -0.0025706, -0.0054199, -0.0026180, -0.0011944, 0.0012616
8: 0.9854277, 0.9874030, 0.9853960, 0.9873697, -0.0008413, 0.0008887
9: -0.0044526, -0.0026595, -0.0044223, -0.0026307, -0.0008067, 0.0007637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005396
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005402
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033353, 0.0042120, 0.0033437, 0.0042289, -0.0004407, 0.0004252
1: 0.0018041, 0.0019308, 0.0018054, 0.0019332, -0.0000637, 0.0000614
2: 0.0120312, 0.0125159, 0.0120218, 0.0125112, -0.0002351, 0.0002437
3: -0.0022373, -0.0017359, -0.0022469, -0.0017408, -0.0002431, 0.0002520
4: -0.0021577, -0.0016150, -0.0021525, -0.0016046, -0.0002728, 0.0002632
5: 0.0056417, 0.0061553, 0.0056318, 0.0061503, -0.0002491, 0.0002582
6: 0.0000842, 0.0021219, 0.0000450, 0.0021024, -0.0009882, 0.0010244
7: -0.0054466, -0.0026713, -0.0054199, -0.0026180, -0.0013951, 0.0013459
8: 0.9853771, 0.9873322, 0.9853960, 0.9873697, -0.0009828, 0.0009481
9: -0.0043882, -0.0026136, -0.0044223, -0.0026307, -0.0008606, 0.0008921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005459, upper bound: 0.0005411
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005411
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033436, 0.0042347, 0.0033423, 0.0042429, -0.0003103, 0.0002998
1: 0.0018054, 0.0019341, 0.0018052, 0.0019353, -0.0000448, 0.0000433
2: 0.0120186, 0.0125113, 0.0120141, 0.0125120, -0.0001658, 0.0001715
3: -0.0022503, -0.0017407, -0.0022549, -0.0017399, -0.0001714, 0.0001774
4: -0.0021525, -0.0016009, -0.0021534, -0.0015958, -0.0001921, 0.0001856
5: 0.0056284, 0.0061504, 0.0056236, 0.0061512, -0.0001756, 0.0001818
6: 0.0000314, 0.0021026, 0.0000123, 0.0021057, -0.0006969, 0.0007212
7: -0.0054202, -0.0025995, -0.0054245, -0.0025735, -0.0009822, 0.0009491
8: 0.9853957, 0.9873828, 0.9853927, 0.9874010, -0.0006919, 0.0006685
9: -0.0044342, -0.0026305, -0.0044508, -0.0026278, -0.0006069, 0.0006280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005459, upper bound: 0.0005444
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005458, upper bound: 0.0005446
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033197, 0.0042054, 0.0033423, 0.0042429, -0.0003889, 0.0003286
1: 0.0018019, 0.0019299, 0.0018052, 0.0019353, -0.0000562, 0.0000475
2: 0.0120348, 0.0125245, 0.0120141, 0.0125120, -0.0001817, 0.0002150
3: -0.0022335, -0.0017270, -0.0022549, -0.0017399, -0.0001879, 0.0002224
4: -0.0021674, -0.0016191, -0.0021534, -0.0015958, -0.0002408, 0.0002034
5: 0.0056455, 0.0061644, 0.0056236, 0.0061512, -0.0001925, 0.0002278
6: 0.0000995, 0.0021582, 0.0000123, 0.0021057, -0.0007638, 0.0009040
7: -0.0054960, -0.0026922, -0.0054245, -0.0025735, -0.0012312, 0.0010402
8: 0.9853423, 0.9873174, 0.9853927, 0.9874010, -0.0008673, 0.0007327
9: -0.0043749, -0.0025820, -0.0044508, -0.0026278, -0.0006651, 0.0007873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005459, upper bound: 0.0005446
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005458, upper bound: 0.0005447
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033228, 0.0042076, 0.0033437, 0.0042289, -0.0003541, 0.0003042
1: 0.0018023, 0.0019302, 0.0018054, 0.0019332, -0.0000512, 0.0000439
2: 0.0120336, 0.0125228, 0.0120218, 0.0125112, -0.0001682, 0.0001958
3: -0.0022348, -0.0017288, -0.0022469, -0.0017408, -0.0001739, 0.0002025
4: -0.0021654, -0.0016177, -0.0021525, -0.0016046, -0.0002192, 0.0001883
5: 0.0056442, 0.0061626, 0.0056318, 0.0061503, -0.0001782, 0.0002074
6: 0.0000943, 0.0021510, 0.0000450, 0.0021024, -0.0007070, 0.0008229
7: -0.0054861, -0.0026851, -0.0054199, -0.0026180, -0.0011208, 0.0009629
8: 0.9853494, 0.9873224, 0.9853960, 0.9873697, -0.0007895, 0.0006783
9: -0.0043794, -0.0025884, -0.0044223, -0.0026307, -0.0006157, 0.0007167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005456, upper bound: 0.0005404
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005458, upper bound: 0.0005415
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0032976, 0.0041798, 0.0033437, 0.0042289, -0.0004169, 0.0003308
1: 0.0017987, 0.0019262, 0.0018054, 0.0019332, -0.0000602, 0.0000478
2: 0.0120489, 0.0125367, 0.0120218, 0.0125112, -0.0001829, 0.0002305
3: -0.0022189, -0.0017144, -0.0022469, -0.0017408, -0.0001891, 0.0002384
4: -0.0021810, -0.0016349, -0.0021525, -0.0016046, -0.0002581, 0.0002047
5: 0.0056605, 0.0061773, 0.0056318, 0.0061503, -0.0001938, 0.0002442
6: 0.0001589, 0.0022096, 0.0000450, 0.0021024, -0.0007688, 0.0009691
7: -0.0055660, -0.0027731, -0.0054199, -0.0026180, -0.0013198, 0.0010470
8: 0.9852931, 0.9872604, 0.9853960, 0.9873697, -0.0009297, 0.0007375
9: -0.0043231, -0.0025373, -0.0044223, -0.0026307, -0.0006695, 0.0008439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005456, upper bound: 0.0005407
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005458, upper bound: 0.0005429
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033778, 0.0042796, 0.0033653, 0.0043160, -0.0004675, 0.0004367
1: 0.0018103, 0.0019406, 0.0018085, 0.0019458, -0.0000675, 0.0000631
2: 0.0119938, 0.0124924, 0.0119736, 0.0124993, -0.0002414, 0.0002584
3: -0.0022759, -0.0017603, -0.0022968, -0.0017531, -0.0002497, 0.0002673
4: -0.0021314, -0.0015731, -0.0021391, -0.0015506, -0.0002894, 0.0002703
5: 0.0056021, 0.0061303, 0.0055807, 0.0061377, -0.0002558, 0.0002738
6: -0.0000729, 0.0020231, -0.0001577, 0.0020521, -0.0010150, 0.0010865
7: -0.0053120, -0.0024574, -0.0053515, -0.0023420, -0.0014797, 0.0013824
8: 0.9854720, 0.9874828, 0.9854442, 0.9875642, -0.0010424, 0.0009738
9: -0.0045250, -0.0026997, -0.0045988, -0.0026744, -0.0008839, 0.0009462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005379
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005379
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033568, 0.0042455, 0.0033653, 0.0043160, -0.0005477, 0.0004646
1: 0.0018073, 0.0019357, 0.0018085, 0.0019458, -0.0000791, 0.0000671
2: 0.0120126, 0.0125040, 0.0119736, 0.0124993, -0.0002569, 0.0003028
3: -0.0022564, -0.0017483, -0.0022968, -0.0017531, -0.0002657, 0.0003132
4: -0.0021443, -0.0015943, -0.0021391, -0.0015506, -0.0003391, 0.0002876
5: 0.0056220, 0.0061426, 0.0055807, 0.0061377, -0.0002722, 0.0003209
6: 0.0000063, 0.0020718, -0.0001577, 0.0020521, -0.0010799, 0.0012731
7: -0.0053783, -0.0025653, -0.0053515, -0.0023420, -0.0017338, 0.0014708
8: 0.9854253, 0.9874068, 0.9854442, 0.9875642, -0.0012213, 0.0010360
9: -0.0044560, -0.0026573, -0.0045988, -0.0026744, -0.0009405, 0.0011087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005381
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005380
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033791, 0.0042651, 0.0033414, 0.0042916, -0.0004686, 0.0004733
1: 0.0018105, 0.0019385, 0.0018050, 0.0019423, -0.0000677, 0.0000684
2: 0.0120018, 0.0124916, 0.0119871, 0.0125125, -0.0002617, 0.0002591
3: -0.0022676, -0.0017610, -0.0022828, -0.0017395, -0.0002706, 0.0002679
4: -0.0021305, -0.0015821, -0.0021539, -0.0015657, -0.0002900, 0.0002930
5: 0.0056106, 0.0061296, 0.0055950, 0.0061517, -0.0002773, 0.0002745
6: -0.0000393, 0.0020200, -0.0001010, 0.0021077, -0.0011001, 0.0010891
7: -0.0053077, -0.0025032, -0.0054272, -0.0024192, -0.0014832, 0.0014982
8: 0.9854749, 0.9874505, 0.9853908, 0.9875097, -0.0010448, 0.0010554
9: -0.0044958, -0.0027024, -0.0045494, -0.0026261, -0.0009580, 0.0009484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005349, upper bound: 0.0005369
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005365, upper bound: 0.0005370
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033582, 0.0042320, 0.0033414, 0.0042916, -0.0005485, 0.0004941
1: 0.0018075, 0.0019337, 0.0018050, 0.0019423, -0.0000792, 0.0000714
2: 0.0120201, 0.0125032, 0.0119871, 0.0125125, -0.0002731, 0.0003033
3: -0.0022487, -0.0017491, -0.0022828, -0.0017395, -0.0002825, 0.0003137
4: -0.0021435, -0.0016026, -0.0021539, -0.0015657, -0.0003395, 0.0003058
5: 0.0056299, 0.0061418, 0.0055950, 0.0061517, -0.0002894, 0.0003213
6: 0.0000377, 0.0020686, -0.0001010, 0.0021077, -0.0011483, 0.0012749
7: -0.0053740, -0.0026080, -0.0054272, -0.0024192, -0.0017363, 0.0015639
8: 0.9854283, 0.9873767, 0.9853908, 0.9875097, -0.0012231, 0.0011016
9: -0.0044287, -0.0026601, -0.0045494, -0.0026261, -0.0010000, 0.0011103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005349, upper bound: 0.0005371
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005365, upper bound: 0.0005371
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033423, 0.0042429, 0.0033653, 0.0043160, -0.0004808, 0.0003768
1: 0.0018052, 0.0019353, 0.0018085, 0.0019458, -0.0000695, 0.0000544
2: 0.0120141, 0.0125120, 0.0119736, 0.0124993, -0.0002083, 0.0002658
3: -0.0022549, -0.0017399, -0.0022968, -0.0017531, -0.0002154, 0.0002749
4: -0.0021534, -0.0015958, -0.0021391, -0.0015506, -0.0002976, 0.0002332
5: 0.0056236, 0.0061512, 0.0055807, 0.0061377, -0.0002207, 0.0002816
6: 0.0000123, 0.0021057, -0.0001577, 0.0020521, -0.0008757, 0.0011175
7: -0.0054245, -0.0025735, -0.0053515, -0.0023420, -0.0015219, 0.0011926
8: 0.9853927, 0.9874010, 0.9854442, 0.9875642, -0.0010721, 0.0008401
9: -0.0044508, -0.0026278, -0.0045988, -0.0026744, -0.0007626, 0.0009731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005498, upper bound: 0.0005546
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005488, upper bound: 0.0005546
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033184, 0.0042138, 0.0033653, 0.0043160, -0.0005592, 0.0004044
1: 0.0018017, 0.0019311, 0.0018085, 0.0019458, -0.0000808, 0.0000584
2: 0.0120302, 0.0125252, 0.0119736, 0.0124993, -0.0002236, 0.0003092
3: -0.0022383, -0.0017263, -0.0022968, -0.0017531, -0.0002313, 0.0003198
4: -0.0021682, -0.0016139, -0.0021391, -0.0015506, -0.0003462, 0.0002504
5: 0.0056406, 0.0061652, 0.0055807, 0.0061377, -0.0002369, 0.0003276
6: 0.0000799, 0.0021612, -0.0001577, 0.0020521, -0.0009400, 0.0012998
7: -0.0055001, -0.0026656, -0.0053515, -0.0023420, -0.0017702, 0.0012802
8: 0.9853395, 0.9873362, 0.9854442, 0.9875642, -0.0012470, 0.0009018
9: -0.0043919, -0.0025794, -0.0045988, -0.0026744, -0.0008186, 0.0011319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005498, upper bound: 0.0005546
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005488, upper bound: 0.0005546
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033437, 0.0042289, 0.0033414, 0.0042916, -0.0004825, 0.0004094
1: 0.0018054, 0.0019332, 0.0018050, 0.0019423, -0.0000697, 0.0000591
2: 0.0120218, 0.0125112, 0.0119871, 0.0125125, -0.0002264, 0.0002668
3: -0.0022469, -0.0017408, -0.0022828, -0.0017395, -0.0002341, 0.0002759
4: -0.0021525, -0.0016046, -0.0021539, -0.0015657, -0.0002987, 0.0002534
5: 0.0056318, 0.0061503, 0.0055950, 0.0061517, -0.0002398, 0.0002827
6: 0.0000450, 0.0021024, -0.0001010, 0.0021077, -0.0009516, 0.0011215
7: -0.0054199, -0.0026180, -0.0054272, -0.0024192, -0.0015274, 0.0012960
8: 0.9853960, 0.9873697, 0.9853908, 0.9875097, -0.0010759, 0.0009129
9: -0.0044223, -0.0026307, -0.0045494, -0.0026261, -0.0008287, 0.0009767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0005545
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005447, upper bound: 0.0005546
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033198, 0.0042002, 0.0033414, 0.0042916, -0.0005606, 0.0004302
1: 0.0018019, 0.0019291, 0.0018050, 0.0019423, -0.0000810, 0.0000622
2: 0.0120377, 0.0125244, 0.0119871, 0.0125125, -0.0002379, 0.0003100
3: -0.0022305, -0.0017271, -0.0022828, -0.0017395, -0.0002460, 0.0003206
4: -0.0021673, -0.0016223, -0.0021539, -0.0015657, -0.0003471, 0.0002663
5: 0.0056486, 0.0061643, 0.0055950, 0.0061517, -0.0002520, 0.0003284
6: 0.0001116, 0.0021579, -0.0001010, 0.0021077, -0.0010000, 0.0013031
7: -0.0054956, -0.0027087, -0.0054272, -0.0024192, -0.0017747, 0.0013619
8: 0.9853426, 0.9873058, 0.9853908, 0.9875097, -0.0012501, 0.0009593
9: -0.0043643, -0.0025823, -0.0045494, -0.0026261, -0.0008708, 0.0011348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0005546
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005447, upper bound: 0.0005546
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033794, 0.0042543, 0.0033187, 0.0042086, -0.0003237, 0.0004712
1: 0.0018105, 0.0019369, 0.0018018, 0.0019303, -0.0000468, 0.0000681
2: 0.0120078, 0.0124915, 0.0120330, 0.0125250, -0.0002605, 0.0001790
3: -0.0022615, -0.0017612, -0.0022353, -0.0017265, -0.0002695, 0.0001851
4: -0.0021304, -0.0015888, -0.0021679, -0.0016171, -0.0002004, 0.0002917
5: 0.0056169, 0.0061294, 0.0056437, 0.0061650, -0.0002760, 0.0001896
6: -0.0000142, 0.0020194, 0.0000920, 0.0021604, -0.0010953, 0.0007523
7: -0.0053070, -0.0025373, -0.0054990, -0.0026820, -0.0010246, 0.0014917
8: 0.9854755, 0.9874265, 0.9853403, 0.9873245, -0.0007217, 0.0010508
9: -0.0044739, -0.0027029, -0.0043814, -0.0025801, -0.0009538, 0.0006551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005444
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005444
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033766, 0.0042538, 0.0033187, 0.0042076, -0.0003361, 0.0004745
1: 0.0018101, 0.0019369, 0.0018018, 0.0019302, -0.0000486, 0.0000686
2: 0.0120080, 0.0124930, 0.0120336, 0.0125251, -0.0002623, 0.0001858
3: -0.0022612, -0.0017596, -0.0022347, -0.0017265, -0.0002713, 0.0001922
4: -0.0021321, -0.0015891, -0.0021680, -0.0016177, -0.0002080, 0.0002937
5: 0.0056172, 0.0061310, 0.0056443, 0.0061650, -0.0002780, 0.0001969
6: -0.0000130, 0.0020258, 0.0000945, 0.0021605, -0.0011029, 0.0007811
7: -0.0053156, -0.0025390, -0.0054991, -0.0026854, -0.0010638, 0.0015021
8: 0.9854693, 0.9874254, 0.9853402, 0.9873223, -0.0007494, 0.0010581
9: -0.0044729, -0.0026974, -0.0043792, -0.0025801, -0.0009605, 0.0006802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005438
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005439
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033585, 0.0042268, 0.0033202, 0.0041949, -0.0003652, 0.0004746
1: 0.0018075, 0.0019330, 0.0018020, 0.0019283, -0.0000528, 0.0000686
2: 0.0120229, 0.0125030, 0.0120406, 0.0125242, -0.0002624, 0.0002019
3: -0.0022458, -0.0017492, -0.0022275, -0.0017273, -0.0002714, 0.0002088
4: -0.0021433, -0.0016058, -0.0021670, -0.0016255, -0.0002261, 0.0002938
5: 0.0056330, 0.0061416, 0.0056517, 0.0061641, -0.0002780, 0.0002140
6: 0.0000496, 0.0020679, 0.0001238, 0.0021570, -0.0011032, 0.0008489
7: -0.0053730, -0.0026243, -0.0054944, -0.0027253, -0.0011561, 0.0015024
8: 0.9854290, 0.9873653, 0.9853435, 0.9872941, -0.0008144, 0.0010583
9: -0.0044183, -0.0026607, -0.0043537, -0.0025831, -0.0009607, 0.0007393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005395
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005402
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033538, 0.0042281, 0.0033201, 0.0041939, -0.0003781, 0.0004790
1: 0.0018068, 0.0019331, 0.0018020, 0.0019282, -0.0000546, 0.0000692
2: 0.0120223, 0.0125056, 0.0120411, 0.0125243, -0.0002648, 0.0002090
3: -0.0022465, -0.0017466, -0.0022269, -0.0017273, -0.0002739, 0.0002162
4: -0.0021462, -0.0016050, -0.0021671, -0.0016262, -0.0002341, 0.0002965
5: 0.0056322, 0.0061444, 0.0056522, 0.0061641, -0.0002806, 0.0002215
6: 0.0000468, 0.0020788, 0.0001261, 0.0021572, -0.0011132, 0.0008788
7: -0.0053879, -0.0026204, -0.0054946, -0.0027285, -0.0011969, 0.0015161
8: 0.9854186, 0.9873680, 0.9853433, 0.9872919, -0.0008431, 0.0010680
9: -0.0044208, -0.0026512, -0.0043517, -0.0025829, -0.0009695, 0.0007653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005402
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005412
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033438, 0.0042216, 0.0033187, 0.0042086, -0.0002948, 0.0003898
1: 0.0018054, 0.0019322, 0.0018018, 0.0019303, -0.0000426, 0.0000563
2: 0.0120259, 0.0125112, 0.0120330, 0.0125250, -0.0002155, 0.0001630
3: -0.0022427, -0.0017408, -0.0022353, -0.0017265, -0.0002229, 0.0001686
4: -0.0021524, -0.0016091, -0.0021679, -0.0016171, -0.0001825, 0.0002413
5: 0.0056361, 0.0061503, 0.0056437, 0.0061650, -0.0002283, 0.0001727
6: 0.0000619, 0.0021021, 0.0000920, 0.0021604, -0.0009059, 0.0006851
7: -0.0054196, -0.0026411, -0.0054990, -0.0026820, -0.0009331, 0.0012338
8: 0.9853963, 0.9873534, 0.9853403, 0.9873245, -0.0006573, 0.0008691
9: -0.0044076, -0.0026309, -0.0043814, -0.0025801, -0.0007889, 0.0005967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005425, upper bound: 0.0005445
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005425, upper bound: 0.0005445
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033429, 0.0042192, 0.0033187, 0.0042076, -0.0003134, 0.0003913
1: 0.0018053, 0.0019318, 0.0018018, 0.0019302, -0.0000453, 0.0000565
2: 0.0120272, 0.0125117, 0.0120336, 0.0125251, -0.0002164, 0.0001733
3: -0.0022414, -0.0017403, -0.0022347, -0.0017265, -0.0002238, 0.0001792
4: -0.0021530, -0.0016105, -0.0021680, -0.0016177, -0.0001940, 0.0002422
5: 0.0056375, 0.0061508, 0.0056443, 0.0061650, -0.0002293, 0.0001836
6: 0.0000675, 0.0021042, 0.0000945, 0.0021605, -0.0009096, 0.0007284
7: -0.0054225, -0.0026486, -0.0054991, -0.0026854, -0.0009920, 0.0012388
8: 0.9853942, 0.9873481, 0.9853402, 0.9873223, -0.0006988, 0.0008726
9: -0.0044028, -0.0026291, -0.0043792, -0.0025801, -0.0007921, 0.0006343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005446
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005447
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033233, 0.0041944, 0.0033202, 0.0041949, -0.0003394, 0.0003939
1: 0.0018024, 0.0019283, 0.0018020, 0.0019283, -0.0000490, 0.0000569
2: 0.0120409, 0.0125225, 0.0120406, 0.0125242, -0.0002178, 0.0001876
3: -0.0022272, -0.0017291, -0.0022275, -0.0017273, -0.0002253, 0.0001940
4: -0.0021651, -0.0016259, -0.0021670, -0.0016255, -0.0002101, 0.0002439
5: 0.0056520, 0.0061623, 0.0056517, 0.0061641, -0.0002308, 0.0001988
6: 0.0001251, 0.0021498, 0.0001238, 0.0021570, -0.0009156, 0.0007888
7: -0.0054845, -0.0027271, -0.0054944, -0.0027253, -0.0010742, 0.0012470
8: 0.9853505, 0.9872928, 0.9853435, 0.9872941, -0.0007567, 0.0008784
9: -0.0043525, -0.0025894, -0.0043537, -0.0025831, -0.0007974, 0.0006869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005425, upper bound: 0.0005404
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005425, upper bound: 0.0005407
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033213, 0.0041926, 0.0033201, 0.0041939, -0.0003581, 0.0003964
1: 0.0018021, 0.0019280, 0.0018020, 0.0019282, -0.0000517, 0.0000573
2: 0.0120419, 0.0125236, 0.0120411, 0.0125243, -0.0002192, 0.0001980
3: -0.0022262, -0.0017280, -0.0022269, -0.0017273, -0.0002267, 0.0002048
4: -0.0021663, -0.0016270, -0.0021671, -0.0016262, -0.0002217, 0.0002454
5: 0.0056530, 0.0061634, 0.0056522, 0.0061641, -0.0002322, 0.0002098
6: 0.0001291, 0.0021543, 0.0001261, 0.0021572, -0.0009214, 0.0008323
7: -0.0054907, -0.0027326, -0.0054946, -0.0027285, -0.0011336, 0.0012549
8: 0.9853461, 0.9872890, 0.9853433, 0.9872919, -0.0007985, 0.0008840
9: -0.0043491, -0.0025854, -0.0043517, -0.0025829, -0.0008024, 0.0007248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005415
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005429
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033780, 0.0042621, 0.0033425, 0.0042815, -0.0004533, 0.0004975
1: 0.0018103, 0.0019381, 0.0018052, 0.0019408, -0.0000655, 0.0000719
2: 0.0120034, 0.0124923, 0.0119928, 0.0125119, -0.0002751, 0.0002506
3: -0.0022659, -0.0017604, -0.0022770, -0.0017401, -0.0002845, 0.0002592
4: -0.0021312, -0.0015839, -0.0021532, -0.0015720, -0.0002806, 0.0003080
5: 0.0056123, 0.0061302, 0.0056010, 0.0061510, -0.0002914, 0.0002655
6: -0.0000324, 0.0020226, -0.0000773, 0.0021051, -0.0011563, 0.0010536
7: -0.0053113, -0.0025126, -0.0054237, -0.0024515, -0.0014349, 0.0015748
8: 0.9854725, 0.9874439, 0.9853933, 0.9874870, -0.0010107, 0.0011093
9: -0.0044897, -0.0027001, -0.0045288, -0.0026283, -0.0010070, 0.0009175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005389
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005391
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033752, 0.0042619, 0.0033424, 0.0042802, -0.0004647, 0.0005002
1: 0.0018099, 0.0019380, 0.0018052, 0.0019407, -0.0000671, 0.0000723
2: 0.0120036, 0.0124938, 0.0119935, 0.0125119, -0.0002766, 0.0002569
3: -0.0022658, -0.0017588, -0.0022763, -0.0017400, -0.0002860, 0.0002657
4: -0.0021330, -0.0015841, -0.0021533, -0.0015728, -0.0002877, 0.0003096
5: 0.0056124, 0.0061319, 0.0056017, 0.0061511, -0.0002930, 0.0002722
6: -0.0000319, 0.0020291, -0.0000743, 0.0021053, -0.0011627, 0.0010802
7: -0.0053202, -0.0025133, -0.0054240, -0.0024555, -0.0014711, 0.0015834
8: 0.9854662, 0.9874434, 0.9853932, 0.9874842, -0.0010363, 0.0011154
9: -0.0044893, -0.0026945, -0.0045262, -0.0026281, -0.0010125, 0.0009407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005379
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005381
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033788, 0.0042556, 0.0033169, 0.0042518, -0.0004480, 0.0005326
1: 0.0018104, 0.0019371, 0.0018015, 0.0019366, -0.0000647, 0.0000769
2: 0.0120070, 0.0124918, 0.0120092, 0.0125261, -0.0002945, 0.0002477
3: -0.0022622, -0.0017608, -0.0022600, -0.0017254, -0.0003045, 0.0002562
4: -0.0021308, -0.0015880, -0.0021691, -0.0015904, -0.0002773, 0.0003297
5: 0.0056161, 0.0061298, 0.0056184, 0.0061660, -0.0003120, 0.0002624
6: -0.0000172, 0.0020208, -0.0000083, 0.0021647, -0.0012379, 0.0010413
7: -0.0053089, -0.0025332, -0.0055049, -0.0025454, -0.0014181, 0.0016859
8: 0.9854742, 0.9874294, 0.9853362, 0.9874209, -0.0009990, 0.0011876
9: -0.0044765, -0.0027017, -0.0044688, -0.0025764, -0.0010780, 0.0009068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005286, upper bound: 0.0005368
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005286, upper bound: 0.0005371
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033787, 0.0042558, 0.0033137, 0.0042509, -0.0004533, 0.0005353
1: 0.0018104, 0.0019371, 0.0018010, 0.0019364, -0.0000655, 0.0000773
2: 0.0120070, 0.0124919, 0.0120096, 0.0125278, -0.0002959, 0.0002506
3: -0.0022623, -0.0017608, -0.0022595, -0.0017236, -0.0003061, 0.0002592
4: -0.0021308, -0.0015879, -0.0021711, -0.0015909, -0.0002806, 0.0003313
5: 0.0056160, 0.0061298, 0.0056189, 0.0061679, -0.0003136, 0.0002655
6: -0.0000176, 0.0020210, -0.0000063, 0.0021721, -0.0012441, 0.0010536
7: -0.0053092, -0.0025328, -0.0055150, -0.0025481, -0.0014349, 0.0016944
8: 0.9854740, 0.9874297, 0.9853290, 0.9874189, -0.0010108, 0.0011935
9: -0.0044768, -0.0027015, -0.0044670, -0.0025699, -0.0010834, 0.0009175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005306, upper bound: 0.0005369
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005306, upper bound: 0.0005371
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033424, 0.0042296, 0.0033425, 0.0042815, -0.0004679, 0.0004276
1: 0.0018052, 0.0019334, 0.0018052, 0.0019408, -0.0000676, 0.0000618
2: 0.0120214, 0.0125119, 0.0119928, 0.0125119, -0.0002364, 0.0002587
3: -0.0022473, -0.0017400, -0.0022770, -0.0017401, -0.0002445, 0.0002675
4: -0.0021533, -0.0016041, -0.0021532, -0.0015720, -0.0002896, 0.0002647
5: 0.0056313, 0.0061511, 0.0056010, 0.0061510, -0.0002505, 0.0002741
6: 0.0000432, 0.0021053, -0.0000773, 0.0021051, -0.0009939, 0.0010874
7: -0.0054240, -0.0026155, -0.0054237, -0.0024515, -0.0014810, 0.0013536
8: 0.9853930, 0.9873714, 0.9853933, 0.9874870, -0.0010432, 0.0009535
9: -0.0044239, -0.0026281, -0.0045288, -0.0026283, -0.0008655, 0.0009470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005448, upper bound: 0.0005520
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005448, upper bound: 0.0005521
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033414, 0.0042274, 0.0033424, 0.0042802, -0.0004777, 0.0004288
1: 0.0018050, 0.0019330, 0.0018052, 0.0019407, -0.0000690, 0.0000620
2: 0.0120227, 0.0125125, 0.0119935, 0.0125119, -0.0002371, 0.0002641
3: -0.0022461, -0.0017395, -0.0022763, -0.0017400, -0.0002452, 0.0002732
4: -0.0021539, -0.0016055, -0.0021533, -0.0015728, -0.0002957, 0.0002654
5: 0.0056327, 0.0061516, 0.0056017, 0.0061511, -0.0002512, 0.0002798
6: 0.0000485, 0.0021076, -0.0000743, 0.0021053, -0.0009967, 0.0011103
7: -0.0054270, -0.0026227, -0.0054240, -0.0024555, -0.0015122, 0.0013574
8: 0.9853910, 0.9873664, 0.9853932, 0.9874842, -0.0010652, 0.0009562
9: -0.0044193, -0.0026262, -0.0045262, -0.0026281, -0.0008680, 0.0009669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005546
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005546
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033439, 0.0042156, 0.0033162, 0.0042593, -0.0004702, 0.0004562
1: 0.0018054, 0.0019313, 0.0018014, 0.0019376, -0.0000679, 0.0000659
2: 0.0120292, 0.0125111, 0.0120050, 0.0125264, -0.0002522, 0.0002600
3: -0.0022393, -0.0017409, -0.0022643, -0.0017250, -0.0002609, 0.0002689
4: -0.0021524, -0.0016128, -0.0021695, -0.0015857, -0.0002911, 0.0002824
5: 0.0056396, 0.0061502, 0.0056139, 0.0061664, -0.0002673, 0.0002754
6: 0.0000758, 0.0021020, -0.0000258, 0.0021663, -0.0010604, 0.0010929
7: -0.0054194, -0.0026600, -0.0055070, -0.0025215, -0.0014884, 0.0014442
8: 0.9853964, 0.9873401, 0.9853346, 0.9874376, -0.0010485, 0.0010174
9: -0.0043955, -0.0026310, -0.0044840, -0.0025750, -0.0009235, 0.0009517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005413, upper bound: 0.0005520
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005413, upper bound: 0.0005521
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033429, 0.0042133, 0.0033160, 0.0042588, -0.0004794, 0.0004571
1: 0.0018053, 0.0019310, 0.0018014, 0.0019376, -0.0000693, 0.0000660
2: 0.0120304, 0.0125117, 0.0120053, 0.0125265, -0.0002527, 0.0002651
3: -0.0022380, -0.0017403, -0.0022640, -0.0017250, -0.0002614, 0.0002741
4: -0.0021530, -0.0016142, -0.0021696, -0.0015860, -0.0002968, 0.0002830
5: 0.0056409, 0.0061508, 0.0056143, 0.0061665, -0.0002678, 0.0002808
6: 0.0000811, 0.0021042, -0.0000245, 0.0021666, -0.0010625, 0.0011143
7: -0.0054224, -0.0026672, -0.0055075, -0.0025233, -0.0015176, 0.0014470
8: 0.9853942, 0.9873350, 0.9853343, 0.9874363, -0.0010690, 0.0010193
9: -0.0043909, -0.0026291, -0.0044829, -0.0025747, -0.0009253, 0.0009704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005414, upper bound: 0.0005546
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005414, upper bound: 0.0005546
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033703, 0.0043040, 0.0033782, 0.0042741, -0.0004055, 0.0004515
1: 0.0018092, 0.0019441, 0.0018103, 0.0019398, -0.0000586, 0.0000652
2: 0.0119803, 0.0124965, 0.0119968, 0.0124922, -0.0002496, 0.0002242
3: -0.0022899, -0.0017560, -0.0022728, -0.0017605, -0.0002582, 0.0002319
4: -0.0021360, -0.0015580, -0.0021311, -0.0015765, -0.0002510, 0.0002795
5: 0.0055878, 0.0061347, 0.0056053, 0.0061301, -0.0002645, 0.0002375
6: -0.0001297, 0.0020405, -0.0000602, 0.0020222, -0.0010494, 0.0009425
7: -0.0053358, -0.0023801, -0.0053108, -0.0024747, -0.0012836, 0.0014292
8: 0.9854553, 0.9875373, 0.9854728, 0.9874706, -0.0009042, 0.0010068
9: -0.0045744, -0.0026845, -0.0045140, -0.0027005, -0.0009139, 0.0008208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005490
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005490
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033676, 0.0043012, 0.0033781, 0.0042736, -0.0004149, 0.0004549
1: 0.0018088, 0.0019437, 0.0018103, 0.0019397, -0.0000599, 0.0000657
2: 0.0119819, 0.0124980, 0.0119971, 0.0124922, -0.0002515, 0.0002294
3: -0.0022883, -0.0017544, -0.0022725, -0.0017604, -0.0002601, 0.0002372
4: -0.0021377, -0.0015598, -0.0021312, -0.0015768, -0.0002568, 0.0002816
5: 0.0055894, 0.0061363, 0.0056056, 0.0061302, -0.0002665, 0.0002430
6: -0.0001231, 0.0020468, -0.0000591, 0.0020224, -0.0010573, 0.0009643
7: -0.0053442, -0.0023891, -0.0053111, -0.0024763, -0.0013132, 0.0014399
8: 0.9854493, 0.9875309, 0.9854727, 0.9874696, -0.0009251, 0.0010143
9: -0.0045687, -0.0026791, -0.0045130, -0.0027003, -0.0009207, 0.0008397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005492
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005492
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033482, 0.0042814, 0.0033787, 0.0042663, -0.0004414, 0.0004529
1: 0.0018060, 0.0019408, 0.0018104, 0.0019387, -0.0000638, 0.0000654
2: 0.0119928, 0.0125087, 0.0120011, 0.0124919, -0.0002504, 0.0002441
3: -0.0022770, -0.0017433, -0.0022683, -0.0017608, -0.0002590, 0.0002524
4: -0.0021497, -0.0015720, -0.0021308, -0.0015814, -0.0002733, 0.0002804
5: 0.0056010, 0.0061477, 0.0056099, 0.0061298, -0.0002653, 0.0002586
6: -0.0000771, 0.0020918, -0.0000421, 0.0020209, -0.0010527, 0.0010260
7: -0.0054056, -0.0024517, -0.0053091, -0.0024994, -0.0013974, 0.0014337
8: 0.9854060, 0.9874868, 0.9854741, 0.9874532, -0.0009843, 0.0010099
9: -0.0045287, -0.0026399, -0.0044981, -0.0027016, -0.0009167, 0.0008935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005344, upper bound: 0.0005359
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005344, upper bound: 0.0005359
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033481, 0.0042801, 0.0033758, 0.0042657, -0.0004445, 0.0004639
1: 0.0018060, 0.0019407, 0.0018100, 0.0019386, -0.0000642, 0.0000670
2: 0.0119935, 0.0125088, 0.0120014, 0.0124935, -0.0002565, 0.0002458
3: -0.0022762, -0.0017433, -0.0022680, -0.0017591, -0.0002653, 0.0002542
4: -0.0021497, -0.0015728, -0.0021326, -0.0015817, -0.0002752, 0.0002872
5: 0.0056018, 0.0061477, 0.0056102, 0.0061315, -0.0002718, 0.0002604
6: -0.0000742, 0.0020920, -0.0000408, 0.0020277, -0.0010782, 0.0010332
7: -0.0054058, -0.0024556, -0.0053182, -0.0025012, -0.0014072, 0.0014685
8: 0.9854059, 0.9874840, 0.9854677, 0.9874520, -0.0009912, 0.0010344
9: -0.0045261, -0.0026397, -0.0044970, -0.0026957, -0.0009390, 0.0008998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005342, upper bound: 0.0005359
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005342, upper bound: 0.0005359
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033467, 0.0042793, 0.0033795, 0.0042597, -0.0004410, 0.0004523
1: 0.0018058, 0.0019405, 0.0018105, 0.0019377, -0.0000637, 0.0000653
2: 0.0119939, 0.0125096, 0.0120048, 0.0124914, -0.0002501, 0.0002438
3: -0.0022758, -0.0017425, -0.0022645, -0.0017612, -0.0002586, 0.0002522
4: -0.0021506, -0.0015733, -0.0021303, -0.0015855, -0.0002730, 0.0002800
5: 0.0056022, 0.0061486, 0.0056137, 0.0061293, -0.0002650, 0.0002583
6: -0.0000723, 0.0020953, -0.0000267, 0.0020191, -0.0010513, 0.0010250
7: -0.0054104, -0.0024582, -0.0053065, -0.0025204, -0.0013959, 0.0014318
8: 0.9854027, 0.9874822, 0.9854758, 0.9874385, -0.0009833, 0.0010086
9: -0.0045245, -0.0026368, -0.0044848, -0.0027032, -0.0009155, 0.0008926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005422
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005422
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033430, 0.0042774, 0.0033794, 0.0042592, -0.0004504, 0.0004567
1: 0.0018053, 0.0019403, 0.0018105, 0.0019376, -0.0000651, 0.0000660
2: 0.0119950, 0.0125116, 0.0120050, 0.0124915, -0.0002525, 0.0002490
3: -0.0022746, -0.0017403, -0.0022643, -0.0017612, -0.0002612, 0.0002575
4: -0.0021529, -0.0015745, -0.0021304, -0.0015857, -0.0002788, 0.0002827
5: 0.0056034, 0.0061507, 0.0056140, 0.0061294, -0.0002675, 0.0002638
6: -0.0000678, 0.0021040, -0.0000256, 0.0020193, -0.0010616, 0.0010468
7: -0.0054222, -0.0024644, -0.0053068, -0.0025218, -0.0014257, 0.0014457
8: 0.9853943, 0.9874780, 0.9854756, 0.9874374, -0.0010043, 0.0010184
9: -0.0045205, -0.0026292, -0.0044838, -0.0027030, -0.0009244, 0.0009116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005448
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005448
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033231, 0.0042517, 0.0033795, 0.0042597, -0.0004848, 0.0004463
1: 0.0018024, 0.0019366, 0.0018105, 0.0019377, -0.0000700, 0.0000645
2: 0.0120092, 0.0125226, 0.0120048, 0.0124914, -0.0002467, 0.0002680
3: -0.0022600, -0.0017290, -0.0022645, -0.0017612, -0.0002552, 0.0002772
4: -0.0021652, -0.0015904, -0.0021303, -0.0015855, -0.0003001, 0.0002763
5: 0.0056184, 0.0061624, 0.0056137, 0.0061293, -0.0002614, 0.0002840
6: -0.0000082, 0.0021501, -0.0000267, 0.0020191, -0.0010373, 0.0011269
7: -0.0054850, -0.0025455, -0.0053065, -0.0025204, -0.0015347, 0.0014127
8: 0.9853501, 0.9874207, 0.9854758, 0.9874385, -0.0010811, 0.0009952
9: -0.0044687, -0.0025891, -0.0044848, -0.0027032, -0.0009033, 0.0009813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005287
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005287
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033193, 0.0042509, 0.0033794, 0.0042592, -0.0004921, 0.0004511
1: 0.0018018, 0.0019364, 0.0018105, 0.0019376, -0.0000711, 0.0000652
2: 0.0120097, 0.0125247, 0.0120050, 0.0124915, -0.0002494, 0.0002721
3: -0.0022595, -0.0017268, -0.0022643, -0.0017612, -0.0002579, 0.0002814
4: -0.0021676, -0.0015909, -0.0021304, -0.0015857, -0.0003046, 0.0002792
5: 0.0056189, 0.0061646, 0.0056140, 0.0061294, -0.0002642, 0.0002883
6: -0.0000062, 0.0021590, -0.0000256, 0.0020193, -0.0010484, 0.0011438
7: -0.0054971, -0.0025482, -0.0053068, -0.0025218, -0.0015578, 0.0014278
8: 0.9853415, 0.9874189, 0.9854756, 0.9874374, -0.0010973, 0.0010058
9: -0.0044669, -0.0025813, -0.0044838, -0.0027030, -0.0009130, 0.0009961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005307
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005307
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033703, 0.0043040, 0.0034073, 0.0043478, -0.0003600, 0.0003191
1: 0.0018092, 0.0019441, 0.0018146, 0.0019504, -0.0000520, 0.0000461
2: 0.0119803, 0.0124965, 0.0119561, 0.0124760, -0.0001764, 0.0001990
3: -0.0022899, -0.0017560, -0.0023149, -0.0017772, -0.0001825, 0.0002058
4: -0.0021360, -0.0015580, -0.0021131, -0.0015309, -0.0002228, 0.0001976
5: 0.0055878, 0.0061347, 0.0055621, 0.0061130, -0.0001870, 0.0002109
6: -0.0001297, 0.0020405, -0.0002314, 0.0019544, -0.0007418, 0.0008367
7: -0.0053358, -0.0023801, -0.0052185, -0.0022415, -0.0011395, 0.0010102
8: 0.9854553, 0.9875373, 0.9855379, 0.9876349, -0.0008027, 0.0007116
9: -0.0045744, -0.0026845, -0.0046631, -0.0027595, -0.0006460, 0.0007286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005430
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005430
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033676, 0.0043012, 0.0034072, 0.0043473, -0.0003724, 0.0003236
1: 0.0018088, 0.0019437, 0.0018145, 0.0019504, -0.0000538, 0.0000468
2: 0.0119819, 0.0124980, 0.0119564, 0.0124761, -0.0001789, 0.0002059
3: -0.0022883, -0.0017544, -0.0023146, -0.0017771, -0.0001850, 0.0002129
4: -0.0021377, -0.0015598, -0.0021131, -0.0015313, -0.0002305, 0.0002003
5: 0.0055894, 0.0061363, 0.0055624, 0.0061131, -0.0001896, 0.0002181
6: -0.0001231, 0.0020468, -0.0002302, 0.0019546, -0.0007521, 0.0008655
7: -0.0053442, -0.0023891, -0.0052187, -0.0022432, -0.0011787, 0.0010243
8: 0.9854493, 0.9875309, 0.9855377, 0.9876337, -0.0008303, 0.0007216
9: -0.0045687, -0.0026791, -0.0046620, -0.0027593, -0.0006550, 0.0007537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005423
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005423
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033482, 0.0042814, 0.0034079, 0.0043400, -0.0004246, 0.0003376
1: 0.0018060, 0.0019408, 0.0018146, 0.0019493, -0.0000613, 0.0000488
2: 0.0119928, 0.0125087, 0.0119604, 0.0124757, -0.0001866, 0.0002347
3: -0.0022770, -0.0017433, -0.0023105, -0.0017775, -0.0001930, 0.0002428
4: -0.0021497, -0.0015720, -0.0021127, -0.0015357, -0.0002628, 0.0002090
5: 0.0056010, 0.0061477, 0.0055667, 0.0061127, -0.0001977, 0.0002487
6: -0.0000771, 0.0020918, -0.0002134, 0.0019531, -0.0007846, 0.0009868
7: -0.0054056, -0.0024517, -0.0052167, -0.0022661, -0.0013439, 0.0010685
8: 0.9854060, 0.9874868, 0.9855391, 0.9876177, -0.0009467, 0.0007527
9: -0.0045287, -0.0026399, -0.0046474, -0.0027607, -0.0006832, 0.0008593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005344, upper bound: 0.0005285
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005344, upper bound: 0.0005285
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033481, 0.0042801, 0.0034046, 0.0043395, -0.0004282, 0.0003481
1: 0.0018060, 0.0019407, 0.0018142, 0.0019492, -0.0000619, 0.0000503
2: 0.0119935, 0.0125088, 0.0119607, 0.0124776, -0.0001925, 0.0002367
3: -0.0022762, -0.0017433, -0.0023102, -0.0017756, -0.0001991, 0.0002448
4: -0.0021497, -0.0015728, -0.0021148, -0.0015361, -0.0002650, 0.0002155
5: 0.0056018, 0.0061477, 0.0055670, 0.0061147, -0.0002039, 0.0002508
6: -0.0000742, 0.0020920, -0.0002122, 0.0019608, -0.0008091, 0.0009952
7: -0.0054058, -0.0024556, -0.0052272, -0.0022678, -0.0013554, 0.0011020
8: 0.9854059, 0.9874840, 0.9855317, 0.9876163, -0.0009548, 0.0007763
9: -0.0045261, -0.0026397, -0.0046463, -0.0027539, -0.0007046, 0.0008667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005342, upper bound: 0.0005285
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005342, upper bound: 0.0005285
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033467, 0.0042793, 0.0034088, 0.0043349, -0.0004015, 0.0003224
1: 0.0018058, 0.0019405, 0.0018148, 0.0019486, -0.0000580, 0.0000466
2: 0.0119939, 0.0125096, 0.0119632, 0.0124752, -0.0001783, 0.0002220
3: -0.0022758, -0.0017425, -0.0023076, -0.0017780, -0.0001844, 0.0002296
4: -0.0021506, -0.0015733, -0.0021122, -0.0015389, -0.0002485, 0.0001996
5: 0.0056022, 0.0061486, 0.0055696, 0.0061122, -0.0001889, 0.0002352
6: -0.0000723, 0.0020953, -0.0002016, 0.0019510, -0.0007495, 0.0009331
7: -0.0054104, -0.0024582, -0.0052138, -0.0022822, -0.0012708, 0.0010207
8: 0.9854027, 0.9874822, 0.9855412, 0.9876063, -0.0008952, 0.0007190
9: -0.0045245, -0.0026368, -0.0046371, -0.0027625, -0.0006527, 0.0008126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005358
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005358
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033430, 0.0042774, 0.0034087, 0.0043345, -0.0004118, 0.0003272
1: 0.0018053, 0.0019403, 0.0018148, 0.0019485, -0.0000595, 0.0000473
2: 0.0119950, 0.0125116, 0.0119634, 0.0124753, -0.0001809, 0.0002277
3: -0.0022746, -0.0017403, -0.0023073, -0.0017779, -0.0001871, 0.0002355
4: -0.0021529, -0.0015745, -0.0021122, -0.0015392, -0.0002549, 0.0002025
5: 0.0056034, 0.0061507, 0.0055699, 0.0061122, -0.0001917, 0.0002412
6: -0.0000678, 0.0021040, -0.0002005, 0.0019512, -0.0007605, 0.0009571
7: -0.0054222, -0.0024644, -0.0052141, -0.0022836, -0.0013034, 0.0010357
8: 0.9853943, 0.9874780, 0.9855409, 0.9876052, -0.0009182, 0.0007296
9: -0.0045205, -0.0026292, -0.0046362, -0.0027623, -0.0006622, 0.0008335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005371
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005371
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0033231, 0.0042517, 0.0034088, 0.0043349, -0.0004614, 0.0003328
1: 0.0018024, 0.0019366, 0.0018148, 0.0019486, -0.0000667, 0.0000481
2: 0.0120092, 0.0125226, 0.0119632, 0.0124752, -0.0001840, 0.0002551
3: -0.0022600, -0.0017290, -0.0023076, -0.0017780, -0.0001903, 0.0002638
4: -0.0021652, -0.0015904, -0.0021122, -0.0015389, -0.0002856, 0.0002060
5: 0.0056184, 0.0061624, 0.0055696, 0.0061122, -0.0001949, 0.0002703
6: -0.0000082, 0.0021501, -0.0002016, 0.0019510, -0.0007735, 0.0010723
7: -0.0054850, -0.0025455, -0.0052138, -0.0022822, -0.0014604, 0.0010534
8: 0.9853501, 0.9874207, 0.9855412, 0.9876063, -0.0010288, 0.0007420
9: -0.0044687, -0.0025891, -0.0046371, -0.0027625, -0.0006736, 0.0009338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005204
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005204
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033193, 0.0042509, 0.0034087, 0.0043345, -0.0004714, 0.0003371
1: 0.0018018, 0.0019364, 0.0018148, 0.0019485, -0.0000681, 0.0000487
2: 0.0120097, 0.0125247, 0.0119634, 0.0124753, -0.0001864, 0.0002606
3: -0.0022595, -0.0017268, -0.0023073, -0.0017779, -0.0001928, 0.0002695
4: -0.0021676, -0.0015909, -0.0021122, -0.0015392, -0.0002918, 0.0002087
5: 0.0056189, 0.0061646, 0.0055699, 0.0061122, -0.0001975, 0.0002761
6: -0.0000062, 0.0021590, -0.0002005, 0.0019512, -0.0007835, 0.0010956
7: -0.0054971, -0.0025482, -0.0052141, -0.0022836, -0.0014922, 0.0010671
8: 0.9853415, 0.9874189, 0.9855409, 0.9876052, -0.0010511, 0.0007517
9: -0.0044669, -0.0025813, -0.0046362, -0.0027623, -0.0006823, 0.0009541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005210
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005210
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0034090, 0.0043286, 0.0033572, 0.0042401, -0.0003580, 0.0005499
1: 0.0018148, 0.0019477, 0.0018073, 0.0019349, -0.0000517, 0.0000794
2: 0.0119667, 0.0124751, 0.0120156, 0.0125038, -0.0003040, 0.0001979
3: -0.0023040, -0.0017781, -0.0022533, -0.0017485, -0.0003144, 0.0002047
4: -0.0021121, -0.0015428, -0.0021441, -0.0015976, -0.0002216, 0.0003404
5: 0.0055734, 0.0061121, 0.0056252, 0.0061424, -0.0003221, 0.0002097
6: -0.0001869, 0.0019506, 0.0000190, 0.0020710, -0.0012781, 0.0008320
7: -0.0052133, -0.0023022, -0.0053772, -0.0025825, -0.0011331, 0.0017406
8: 0.9855415, 0.9875922, 0.9854261, 0.9873947, -0.0007982, 0.0012261
9: -0.0046243, -0.0027628, -0.0044450, -0.0026580, -0.0011130, 0.0007246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005355
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005355
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033658, 0.0043034, 0.0033572, 0.0042401, -0.0004193, 0.0005424
1: 0.0018086, 0.0019440, 0.0018073, 0.0019349, -0.0000606, 0.0000784
2: 0.0119806, 0.0124990, 0.0120156, 0.0125038, -0.0002999, 0.0002318
3: -0.0022895, -0.0017534, -0.0022533, -0.0017485, -0.0003102, 0.0002397
4: -0.0021388, -0.0015584, -0.0021441, -0.0015976, -0.0002595, 0.0003358
5: 0.0055882, 0.0061374, 0.0056252, 0.0061424, -0.0003177, 0.0002456
6: -0.0001282, 0.0020509, 0.0000190, 0.0020710, -0.0012607, 0.0009745
7: -0.0053499, -0.0023821, -0.0053772, -0.0025825, -0.0013271, 0.0017170
8: 0.9854453, 0.9875358, 0.9854261, 0.9873947, -0.0009349, 0.0012095
9: -0.0045731, -0.0026755, -0.0044450, -0.0026580, -0.0010979, 0.0008486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005355
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005355
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0034056, 0.0043276, 0.0033571, 0.0042396, -0.0003671, 0.0005527
1: 0.0018143, 0.0019475, 0.0018073, 0.0019348, -0.0000530, 0.0000798
2: 0.0119673, 0.0124770, 0.0120159, 0.0125038, -0.0003056, 0.0002030
3: -0.0023034, -0.0017762, -0.0022530, -0.0017484, -0.0003160, 0.0002099
4: -0.0021141, -0.0015434, -0.0021442, -0.0015979, -0.0002273, 0.0003421
5: 0.0055740, 0.0061140, 0.0056255, 0.0061425, -0.0003238, 0.0002151
6: -0.0001845, 0.0019584, 0.0000201, 0.0020711, -0.0012846, 0.0008534
7: -0.0052238, -0.0023054, -0.0053774, -0.0025841, -0.0011622, 0.0017495
8: 0.9855341, 0.9875899, 0.9854259, 0.9873935, -0.0008187, 0.0012324
9: -0.0046222, -0.0027561, -0.0044440, -0.0026579, -0.0011187, 0.0007431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005359
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005359
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0033651, 0.0043012, 0.0033571, 0.0042396, -0.0004280, 0.0005464
1: 0.0018085, 0.0019437, 0.0018073, 0.0019348, -0.0000618, 0.0000789
2: 0.0119819, 0.0124994, 0.0120159, 0.0125038, -0.0003021, 0.0002366
3: -0.0022883, -0.0017530, -0.0022530, -0.0017484, -0.0003124, 0.0002447
4: -0.0021393, -0.0015598, -0.0021442, -0.0015979, -0.0002649, 0.0003382
5: 0.0055894, 0.0061378, 0.0056255, 0.0061425, -0.0003201, 0.0002507
6: -0.0001231, 0.0020527, 0.0000201, 0.0020711, -0.0012699, 0.0009947
7: -0.0053523, -0.0023890, -0.0053774, -0.0025841, -0.0013547, 0.0017295
8: 0.9854436, 0.9875310, 0.9854259, 0.9873935, -0.0009543, 0.0012183
9: -0.0045687, -0.0026739, -0.0044440, -0.0026579, -0.0011059, 0.0008663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005359
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005359
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033851, 0.0043041, 0.0033586, 0.0042264, -0.0003914, 0.0005496
1: 0.0018114, 0.0019441, 0.0018075, 0.0019329, -0.0000566, 0.0000794
2: 0.0119802, 0.0124883, 0.0120232, 0.0125030, -0.0003038, 0.0002164
3: -0.0022899, -0.0017645, -0.0022455, -0.0017493, -0.0003142, 0.0002238
4: -0.0021268, -0.0015580, -0.0021433, -0.0016061, -0.0002423, 0.0003402
5: 0.0055877, 0.0061260, 0.0056332, 0.0061416, -0.0003219, 0.0002293
6: -0.0001299, 0.0020060, 0.0000507, 0.0020678, -0.0012773, 0.0009098
7: -0.0052887, -0.0023797, -0.0053729, -0.0026258, -0.0012391, 0.0017396
8: 0.9854884, 0.9875375, 0.9854292, 0.9873642, -0.0008728, 0.0012254
9: -0.0045747, -0.0027146, -0.0044173, -0.0026608, -0.0011123, 0.0007923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005267, upper bound: 0.0005286
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0005267, upper bound: 0.0005287
time: 0.69 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.13 seconds
IS_A1_B1_B1_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005531
IS_A1_B1_B1_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005530
IS_A1_B1_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005538
IS_A1_B1_B1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005538
IS_A1_B1_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005417, upper bound: 0.0005469
IS_A1_B1_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005417, upper bound: 0.0005469
IS_A1_B1_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005469
IS_A1_B1_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005469
IS_A1_B1_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005474
IS_A1_B1_B1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005474
IS_A1_B1_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005498
IS_A1_B1_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005498
IS_A1_B1_B1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005402
IS_A1_B1_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005402
IS_A1_B1_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005436
IS_A1_B1_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005419, upper bound: 0.0005436
IS_A1_B1_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005443, upper bound: 0.0005475
IS_A1_B1_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005443, upper bound: 0.0005475
IS_A1_B1_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005444, upper bound: 0.0005478
IS_A1_B1_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005444, upper bound: 0.0005478
IS_A1_B1_B1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005409, upper bound: 0.0005281
IS_A1_B1_B1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005409, upper bound: 0.0005281
IS_A1_B1_B1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005403, upper bound: 0.0005281
IS_A1_B1_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005403, upper bound: 0.0005281
IS_A1_B1_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005478
IS_A1_B1_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005478
IS_A1_B1_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005400, upper bound: 0.0005478
IS_A1_B1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005400, upper bound: 0.0005478
IS_A1_B1_B1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
IS_A1_B1_B1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
IS_A1_B1_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
IS_A1_B1_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005325, upper bound: 0.0005269
IS_A1_B1_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005460
IS_A1_B1_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005460
IS_A1_B1_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005460
IS_A1_B1_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005460
IS_A1_B1_B2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005469
IS_A1_B1_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005469
IS_A1_B1_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005469
IS_A1_B1_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005469
IS_A1_B1_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005402
IS_A1_B1_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005402
IS_A1_B1_B2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005402
IS_A1_B1_B2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005402
IS_A1_B1_B2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005433
IS_A1_B1_B2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005436
IS_A1_B1_B2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005433
IS_A1_B1_B2_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005388, upper bound: 0.0005436
IS_A1_B1_B2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005265
IS_A1_B1_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005265
IS_A1_B1_B2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005265
IS_A1_B1_B2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005265
IS_A1_B1_B2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005278
IS_A1_B1_B2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005281
IS_A1_B1_B2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005278
IS_A1_B1_B2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005334, upper bound: 0.0005281
IS_A1_B1_B2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005256
IS_A1_B1_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005256
IS_A1_B1_B2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005256
IS_A1_B1_B2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005256
IS_A1_B1_B2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005267
IS_A1_B1_B2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005269
IS_A1_B1_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005267
IS_A1_B1_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005277, upper bound: 0.0005269
IS_A1_B2_B1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005453, upper bound: 0.0005437
IS_A1_B2_B1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005437
IS_A1_B2_B1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005453, upper bound: 0.0005438
IS_A1_B2_B1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005439
IS_A1_B2_B1_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005396
IS_A1_B2_B1_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005402
IS_A1_B2_B1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005459, upper bound: 0.0005411
IS_A1_B2_B1_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005489, upper bound: 0.0005411
IS_A1_B2_B1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005459, upper bound: 0.0005444
IS_A1_B2_B1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005458, upper bound: 0.0005446
IS_A1_B2_B1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005459, upper bound: 0.0005446
IS_A1_B2_B1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005458, upper bound: 0.0005447
IS_A1_B2_B1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005456, upper bound: 0.0005404
IS_A1_B2_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005458, upper bound: 0.0005415
IS_A1_B2_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005456, upper bound: 0.0005407
IS_A1_B2_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005458, upper bound: 0.0005429
IS_A1_B2_B1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005379
IS_A1_B2_B1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005379
IS_A1_B2_B1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005423, upper bound: 0.0005381
IS_A1_B2_B1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005380
IS_A1_B2_B1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005349, upper bound: 0.0005369
IS_A1_B2_B1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005365, upper bound: 0.0005370
IS_A1_B2_B1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005349, upper bound: 0.0005371
IS_A1_B2_B1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005365, upper bound: 0.0005371
IS_A1_B2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005498, upper bound: 0.0005546
IS_A1_B2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005488, upper bound: 0.0005546
IS_A1_B2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005498, upper bound: 0.0005546
IS_A1_B2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005488, upper bound: 0.0005546
IS_A1_B2_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0005545
IS_A1_B2_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005447, upper bound: 0.0005546
IS_A1_B2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005449, upper bound: 0.0005546
IS_A1_B2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005447, upper bound: 0.0005546
IS_A1_B2_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005444
IS_A1_B2_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005444
IS_A1_B2_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005438
IS_A1_B2_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005439
IS_A1_B2_B2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005395
IS_A1_B2_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005402
IS_A1_B2_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005402
IS_A1_B2_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005433, upper bound: 0.0005412
IS_A1_B2_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005425, upper bound: 0.0005445
IS_A1_B2_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005425, upper bound: 0.0005445
IS_A1_B2_B2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005446
IS_A1_B2_B2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005447
IS_A1_B2_B2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005425, upper bound: 0.0005404
IS_A1_B2_B2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005425, upper bound: 0.0005407
IS_A1_B2_B2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005415
IS_A1_B2_B2_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005426, upper bound: 0.0005429
IS_A1_B2_B2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005389
IS_A1_B2_B2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005391
IS_A1_B2_B2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005379
IS_A1_B2_B2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005359, upper bound: 0.0005381
IS_A1_B2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005286, upper bound: 0.0005368
IS_A1_B2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005286, upper bound: 0.0005371
IS_A1_B2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005306, upper bound: 0.0005369
IS_A1_B2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005306, upper bound: 0.0005371
IS_A1_B2_B2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005448, upper bound: 0.0005520
IS_A1_B2_B2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005448, upper bound: 0.0005521
IS_A1_B2_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005546
IS_A1_B2_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005450, upper bound: 0.0005546
IS_A1_B2_B2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005413, upper bound: 0.0005520
IS_A1_B2_B2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005413, upper bound: 0.0005521
IS_A1_B2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005414, upper bound: 0.0005546
IS_A1_B2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005414, upper bound: 0.0005546
IS_A2_B1_B1_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005490
IS_A2_B1_B1_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005490
IS_A2_B1_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005492
IS_A2_B1_B1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005492
IS_A2_B1_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005344, upper bound: 0.0005359
IS_A2_B1_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005344, upper bound: 0.0005359
IS_A2_B1_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005342, upper bound: 0.0005359
IS_A2_B1_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005342, upper bound: 0.0005359
IS_A2_B1_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005422
IS_A2_B1_B1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005422
IS_A2_B1_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005448
IS_A2_B1_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005448
IS_A2_B1_B1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005287
IS_A2_B1_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005287
IS_A2_B1_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005307
IS_A2_B1_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005307
IS_A2_B1_B1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005430
IS_A2_B1_B1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005430
IS_A2_B1_B1_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005423
IS_A2_B1_B1_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005423
IS_A2_B1_B1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005344, upper bound: 0.0005285
IS_A2_B1_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005344, upper bound: 0.0005285
IS_A2_B1_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005342, upper bound: 0.0005285
IS_A2_B1_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005342, upper bound: 0.0005285
IS_A2_B1_B1_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005358
IS_A2_B1_B1_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005358
IS_A2_B1_B1_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005371
IS_A2_B1_B1_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005478, upper bound: 0.0005371
IS_A2_B1_B1_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005204
IS_A2_B1_B1_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005204
IS_A2_B1_B1_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005210
IS_A2_B1_B1_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005339, upper bound: 0.0005210
IS_A2_B1_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005355
IS_A2_B1_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005355
IS_A2_B1_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005355
IS_A2_B1_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005355
IS_A2_B1_B2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005359
IS_A2_B1_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005359
IS_A2_B1_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005359
IS_A2_B1_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005278, upper bound: 0.0005359
IS_A2_B1_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005267, upper bound: 0.0005286
IS_A2_B1_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.13
Output dim: 8, lower bound: -0.0005267, upper bound: 0.0005287
IS_A2_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005287
IS_A2_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005307
IS_A2_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005307
IS_A2_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005289
IS_A2_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005289
IS_A2_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005285
IS_A2_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005285
IS_A2_B1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005204
IS_A2_B1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005204
IS_A2_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005210
IS_A2_B1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005210
IS_A2_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005568
IS_A2_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005396, upper bound: 0.0005568
IS_A2_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005485
IS_A2_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005485
IS_A2_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005624
IS_A2_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005624
IS_A2_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005575
IS_A2_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005575
IS_A2_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005437
IS_A2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005482, upper bound: 0.0005437
IS_A2_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005437
IS_A2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005386, upper bound: 0.0005437
IS_A2_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005551
IS_A2_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005748, upper bound: 0.0005551
IS_A2_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005551
IS_A2_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005707, upper bound: 0.0005551
IS_A2_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005265, upper bound: 0.0005378
IS_A2_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005281, upper bound: 0.0005379
IS_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005256, upper bound: 0.0005325
IS_A2_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005325
IS_A2_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005437
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005438
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005413
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005414
IS_A2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005354, upper bound: 0.0005282
IS_A2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005354, upper bound: 0.0005282
IS_A2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005256, upper bound: 0.0005279
IS_A2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005269, upper bound: 0.0005279
IS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005614, upper bound: 0.0005373
IS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005611, upper bound: 0.0005374
IS_A2_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005579, upper bound: 0.0005373
IS_A2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.13
Output dim: 8, lower bound: -0.0005577, upper bound: 0.0005374

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.78 + 597.71 = 600.49 seconds
