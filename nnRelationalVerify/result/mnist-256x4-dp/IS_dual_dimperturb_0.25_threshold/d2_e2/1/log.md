## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.10584259


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0479110, 0.0701729, -0.0479110, 0.0701729, -0.1180838, 0.1180838)
1: (-0.0172781, 0.0171990, -0.0172781, 0.0171990, -0.0344771, 0.0344771)
2: (-0.0060776, 0.0407657, -0.0060776, 0.0407657, -0.0468434, 0.0468434)
3: (-0.0081011, 0.0744717, -0.0081011, 0.0744717, -0.0748520, 0.0748521)
4: (-0.0261970, -0.0005017, -0.0261970, -0.0005017, -0.0256953, 0.0256953)
5: (0.0016153, 0.0518840, 0.0016153, 0.0518840, -0.0492261, 0.0492261)
6: (-0.0383034, 0.0606607, -0.0383034, 0.0606607, -0.0989641, 0.0989641)
7: (-0.0188207, 0.0142380, -0.0188207, 0.0142380, -0.0330586, 0.0330586)
8: (0.6702957, 0.9512489, 0.6702957, 0.9512489, -0.2350330, 0.2350333)
9: (0.0458664, 0.0959938, 0.0458664, 0.0959938, -0.0501274, 0.0501274)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.37 = 2.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1373574, upper bound: 0.1373574

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274570, upper bound: 0.1348970
time: 0.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354609, upper bound: 0.1354609
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.18 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 8, lower bound: -0.1274570, upper bound: 0.1348970
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.18
Output dim: 8, lower bound: -0.1354609, upper bound: 0.1354609

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0510144, 0.0648657, -0.0478677, 0.0692620, -0.1202764, 0.1127334
1: -0.0190375, 0.0191981, -0.0172537, 0.0171715, -0.0362090, 0.0364518
2: -0.0082250, 0.0389526, -0.0060478, 0.0404545, -0.0486795, 0.0450004
3: -0.0107923, 0.0710048, -0.0080635, 0.0738765, -0.0762836, 0.0711831
4: -0.0271825, 0.0008495, -0.0261834, -0.0005204, -0.0266621, 0.0270329
5: -0.0009829, 0.0500243, 0.0016515, 0.0515649, -0.0518383, 0.0483728
6: -0.0410436, 0.0554467, -0.0382651, 0.0597657, -0.1008093, 0.0937117
7: -0.0207957, 0.0156302, -0.0187933, 0.0142187, -0.0350144, 0.0344235
8: 0.6852283, 0.9560106, 0.6728577, 0.9511824, -0.2189879, 0.2333884
9: 0.0432520, 0.0942116, 0.0459026, 0.0956879, -0.0524359, 0.0483090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0977875, upper bound: 0.1295554
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1036655, upper bound: 0.1175611
time: 0.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0478881, 0.0697145, -0.0479110, 0.0701729, -0.1180610, 0.1176255
1: -0.0172652, 0.0171843, -0.0172781, 0.0171990, -0.0344642, 0.0344624
2: -0.0060618, 0.0406091, -0.0060776, 0.0407657, -0.0468276, 0.0466868
3: -0.0080812, 0.0741722, -0.0081011, 0.0744717, -0.0748326, 0.0743098
4: -0.0261897, -0.0005115, -0.0261970, -0.0005017, -0.0256881, 0.0256854
5: 0.0016343, 0.0517235, 0.0016153, 0.0518840, -0.0492069, 0.0491117
6: -0.0382832, 0.0602105, -0.0383034, 0.0606607, -0.0989439, 0.0985139
7: -0.0188062, 0.0142278, -0.0188207, 0.0142380, -0.0330441, 0.0330485
8: 0.6715845, 0.9512134, 0.6702957, 0.9512489, -0.2322617, 0.2350004
9: 0.0458857, 0.0958399, 0.0458664, 0.0959938, -0.0501081, 0.0499735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1270266, upper bound: 0.1265451
time: 0.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1270266, upper bound: 0.1270266
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.39 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 8, lower bound: -0.0977875, upper bound: 0.1295554
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 8, lower bound: -0.1036655, upper bound: 0.1175611
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 8, lower bound: -0.1270266, upper bound: 0.1265451
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 8, lower bound: -0.1270266, upper bound: 0.1270266

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0510148, 0.0648533, -0.0478681, 0.0676690, -0.1186838, 0.1127213
1: -0.0190376, 0.0191981, -0.0172537, 0.0171715, -0.0362091, 0.0364518
2: -0.0082249, 0.0389483, -0.0060478, 0.0399103, -0.0481351, 0.0449961
3: -0.0107924, 0.0709966, -0.0080635, 0.0728360, -0.0751894, 0.0711750
4: -0.0271825, 0.0008495, -0.0261833, -0.0005204, -0.0266621, 0.0270329
5: -0.0009829, 0.0500200, 0.0016514, 0.0510066, -0.0512188, 0.0483686
6: -0.0410436, 0.0554346, -0.0382651, 0.0582007, -0.0992443, 0.0936997
7: -0.0207958, 0.0156302, -0.0187933, 0.0142187, -0.0350145, 0.0344235
8: 0.6852626, 0.9560105, 0.6773410, 0.9511834, -0.2189541, 0.2286196
9: 0.0432518, 0.0942075, 0.0459027, 0.0951529, -0.0519011, 0.0483048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.47 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
time: 0.48 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0510145, 0.0641234, -0.0432868, 0.0632949, -0.1143094, 0.1074102
1: -0.0190375, 0.0191981, -0.0146568, 0.0142207, -0.0332582, 0.0338548
2: -0.0082251, 0.0386989, -0.0028782, 0.0384159, -0.0466409, 0.0415771
3: -0.0107923, 0.0705199, -0.0040909, 0.0699787, -0.0726909, 0.0667738
4: -0.0271825, 0.0008495, -0.0247287, -0.0025147, -0.0246678, 0.0255782
5: -0.0009828, 0.0497642, 0.0054865, 0.0494739, -0.0479740, 0.0442777
6: -0.0410434, 0.0547176, -0.0342208, 0.0539036, -0.0949470, 0.0889384
7: -0.0207957, 0.0156302, -0.0158781, 0.0121640, -0.0329597, 0.0315083
8: 0.6873170, 0.9560101, 0.6896484, 0.9441544, -0.2102561, 0.2185009
9: 0.0432518, 0.0939624, 0.0497616, 0.0936842, -0.0504324, 0.0442008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1086641
time: 0.54 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0691003, -0.0483787, 0.0659988, -0.1138870, 0.1174790
1: -0.0172653, 0.0171843, -0.0175432, 0.0175002, -0.0347655, 0.0347274
2: -0.0060618, 0.0403992, -0.0064009, 0.0393397, -0.0454015, 0.0468001
3: -0.0080810, 0.0737708, -0.0085063, 0.0717448, -0.0719549, 0.0740180
4: -0.0261897, -0.0005116, -0.0263454, -0.0002981, -0.0258916, 0.0258339
5: 0.0016344, 0.0515081, 0.0012239, 0.0504213, -0.0487869, 0.0492357
6: -0.0382837, 0.0596068, -0.0387159, 0.0565598, -0.0948434, 0.0983227
7: -0.0188062, 0.0142279, -0.0191182, 0.0144477, -0.0332539, 0.0333461
8: 0.6733133, 0.9512133, 0.6820396, 0.9519657, -0.2291915, 0.2223325
9: 0.0458857, 0.0956335, 0.0454723, 0.0945921, -0.0487065, 0.0501612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
time: 0.50 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144414
time: 0.51 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0478881, 0.0697145, -0.0479110, 0.0671977, -0.1150858, 0.1176255
1: -0.0172652, 0.0171843, -0.0172780, 0.0171990, -0.0344642, 0.0344623
2: -0.0060618, 0.0406091, -0.0060776, 0.0397492, -0.0458110, 0.0466867
3: -0.0080812, 0.0741722, -0.0081008, 0.0725281, -0.0726549, 0.0743095
4: -0.0261897, -0.0005115, -0.0261969, -0.0005017, -0.0256881, 0.0256853
5: 0.0016343, 0.0517235, 0.0016153, 0.0508415, -0.0485761, 0.0491114
6: -0.0382832, 0.0602105, -0.0383031, 0.0577376, -0.0960208, 0.0985135
7: -0.0188062, 0.0142278, -0.0188207, 0.0142381, -0.0330442, 0.0330485
8: 0.6715845, 0.9512134, 0.6786667, 0.9512488, -0.2322607, 0.2248352
9: 0.0458857, 0.0958399, 0.0458664, 0.0949947, -0.0491090, 0.0499735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1149167
time: 0.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.39 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1086641
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144414
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1149167

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0510147, 0.0647588, -0.0483328, 0.0637884, -0.1148030, 0.1130916
1: -0.0190376, 0.0191981, -0.0175171, 0.0174706, -0.0365082, 0.0367152
2: -0.0082249, 0.0389161, -0.0063692, 0.0385845, -0.0468094, 0.0452852
3: -0.0107923, 0.0709348, -0.0084666, 0.0703007, -0.0725117, 0.0711887
4: -0.0271825, 0.0008495, -0.0263309, -0.0003181, -0.0268643, 0.0271804
5: -0.0009829, 0.0499868, 0.0012626, 0.0496467, -0.0506296, 0.0487242
6: -0.0410434, 0.0553416, -0.0386754, 0.0543882, -0.0954316, 0.0940170
7: -0.0207958, 0.0156302, -0.0190889, 0.0144272, -0.0352230, 0.0347191
8: 0.6855294, 0.9560105, 0.6882600, 0.9518959, -0.2172661, 0.2168026
9: 0.0432519, 0.0941757, 0.0455112, 0.0938498, -0.0505979, 0.0486645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.49 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0510144, 0.0648657, -0.0478678, 0.0645912, -0.1156056, 0.1127336
1: -0.0190375, 0.0191981, -0.0172537, 0.0171715, -0.0362090, 0.0364518
2: -0.0082250, 0.0389526, -0.0060478, 0.0388587, -0.0470838, 0.0450003
3: -0.0107923, 0.0710048, -0.0080637, 0.0708254, -0.0729987, 0.0711828
4: -0.0271825, 0.0008495, -0.0261833, -0.0005204, -0.0266621, 0.0270329
5: -0.0009829, 0.0500243, 0.0016516, 0.0499281, -0.0506250, 0.0483727
6: -0.0410436, 0.0554467, -0.0382654, 0.0551771, -0.0962207, 0.0937121
7: -0.0207957, 0.0156302, -0.0187933, 0.0142187, -0.0350144, 0.0344235
8: 0.6852283, 0.9560106, 0.6860011, 0.9511826, -0.2189865, 0.2183182
9: 0.0432520, 0.0942116, 0.0459027, 0.0941195, -0.0508675, 0.0483089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0510147, 0.0647588, -0.0436411, 0.0592813, -0.1102960, 0.1083999
1: -0.0190376, 0.0191981, -0.0148572, 0.0144485, -0.0334861, 0.0340553
2: -0.0082249, 0.0389161, -0.0031229, 0.0370446, -0.0452695, 0.0420389
3: -0.0107923, 0.0709348, -0.0043976, 0.0673565, -0.0699745, 0.0672492
4: -0.0271825, 0.0008495, -0.0248410, -0.0023608, -0.0248216, 0.0256906
5: -0.0009829, 0.0499868, 0.0051905, 0.0480673, -0.0476803, 0.0447963
6: -0.0410434, 0.0553416, -0.0345330, 0.0499602, -0.0910036, 0.0898746
7: -0.0207958, 0.0156302, -0.0161031, 0.0123225, -0.0331183, 0.0317333
8: 0.6855294, 0.9560105, 0.7009428, 0.9446972, -0.2109051, 0.2065475
9: 0.0432519, 0.0941757, 0.0494637, 0.0923364, -0.0490845, 0.0447120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0510144, 0.0648657, -0.0432869, 0.0601637, -0.1111781, 0.1081526
1: -0.0190375, 0.0191981, -0.0146567, 0.0142207, -0.0332583, 0.0338549
2: -0.0082250, 0.0389526, -0.0028783, 0.0373461, -0.0455711, 0.0418309
3: -0.0107923, 0.0710048, -0.0040907, 0.0679330, -0.0705031, 0.0672598
4: -0.0271825, 0.0008495, -0.0247287, -0.0025147, -0.0246677, 0.0255782
5: -0.0009829, 0.0500243, 0.0054865, 0.0483766, -0.0473337, 0.0445378
6: -0.0410436, 0.0554467, -0.0342208, 0.0508273, -0.0918709, 0.0896675
7: -0.0207957, 0.0156302, -0.0158781, 0.0121639, -0.0329596, 0.0315083
8: 0.6852283, 0.9560106, 0.6984591, 0.9441545, -0.2123556, 0.2082403
9: 0.0432520, 0.0942116, 0.0497617, 0.0926327, -0.0493808, 0.0444500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1039127
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0695980, -0.0483788, 0.0643713, -0.1122595, 0.1179768
1: -0.0172652, 0.0171843, -0.0175432, 0.0175002, -0.0347654, 0.0347275
2: -0.0060618, 0.0405693, -0.0064011, 0.0387837, -0.0448454, 0.0469704
3: -0.0080810, 0.0740961, -0.0085063, 0.0706818, -0.0708401, 0.0743525
4: -0.0261897, -0.0005115, -0.0263454, -0.0002981, -0.0258916, 0.0258338
5: 0.0016344, 0.0516826, 0.0012239, 0.0498511, -0.0482167, 0.0492716
6: -0.0382833, 0.0600959, -0.0387161, 0.0549611, -0.0932444, 0.0988120
7: -0.0188062, 0.0142279, -0.0191182, 0.0144478, -0.0332539, 0.0333461
8: 0.6719122, 0.9512145, 0.6866189, 0.9519659, -0.2306788, 0.2174637
9: 0.0458855, 0.0958008, 0.0454726, 0.0940456, -0.0481601, 0.0503282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
time: 0.51 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0478880, 0.0692195, -0.0436862, 0.0600004, -0.1078883, 0.1129057
1: -0.0172652, 0.0171843, -0.0148830, 0.0144780, -0.0317432, 0.0320673
2: -0.0060619, 0.0404400, -0.0031544, 0.0372902, -0.0433521, 0.0435944
3: -0.0080812, 0.0738488, -0.0044372, 0.0678261, -0.0683597, 0.0701385
4: -0.0261897, -0.0005115, -0.0248554, -0.0023410, -0.0238487, 0.0243439
5: 0.0016344, 0.0515499, 0.0051522, 0.0483193, -0.0449903, 0.0454371
6: -0.0382830, 0.0597239, -0.0345731, 0.0506667, -0.0889498, 0.0942970
7: -0.0188062, 0.0142279, -0.0161321, 0.0123429, -0.0311491, 0.0303601
8: 0.6729783, 0.9512145, 0.6989187, 0.9447667, -0.2231350, 0.2073948
9: 0.0458855, 0.0956736, 0.0494252, 0.0925778, -0.0466924, 0.0462484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144414
time: 0.50 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144414
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0479110, 0.0671977, -0.1150859, 0.1159527
1: -0.0172653, 0.0171843, -0.0172780, 0.0171990, -0.0344643, 0.0344623
2: -0.0060618, 0.0400376, -0.0060776, 0.0397492, -0.0458111, 0.0461152
3: -0.0080811, 0.0730793, -0.0081008, 0.0725281, -0.0726549, 0.0731865
4: -0.0261897, -0.0005116, -0.0261969, -0.0005017, -0.0256880, 0.0256853
5: 0.0016343, 0.0511372, 0.0016153, 0.0508415, -0.0485761, 0.0484890
6: -0.0382830, 0.0585671, -0.0383031, 0.0577376, -0.0960205, 0.0968701
7: -0.0188062, 0.0142279, -0.0188207, 0.0142381, -0.0330442, 0.0330486
8: 0.6762923, 0.9512142, 0.6786667, 0.9512488, -0.2273870, 0.2248356
9: 0.0458855, 0.0952781, 0.0458664, 0.0949947, -0.0491092, 0.0494117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479108, 0.0662074, -0.1095146, 0.1116857
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0394109, -0.0423031, 0.0446573
3: -0.0041084, 0.0702920, -0.0081008, 0.0718812, -0.0680928, 0.0706359
4: -0.0247350, -0.0024921, -0.0261969, -0.0005017, -0.0242333, 0.0237048
5: 0.0054697, 0.0496420, 0.0016155, 0.0504945, -0.0447388, 0.0452268
6: -0.0342384, 0.0543750, -0.0383035, 0.0567648, -0.0910032, 0.0926785
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6814528, 0.9512495, -0.2168379, 0.2154725
9: 0.0497448, 0.0938453, 0.0458663, 0.0946622, -0.0449174, 0.0479790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144016
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1149167
time: 0.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.33 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
IS_A1_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1039127
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144414
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144414
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144016
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1149167

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0510148, 0.0648533, -0.0483328, 0.0637884, -0.1148032, 0.1131861
1: -0.0190376, 0.0191981, -0.0175171, 0.0174706, -0.0365082, 0.0367152
2: -0.0082249, 0.0389483, -0.0063692, 0.0385845, -0.0468094, 0.0453174
3: -0.0107924, 0.0709966, -0.0084666, 0.0703007, -0.0725119, 0.0712514
4: -0.0271825, 0.0008495, -0.0263309, -0.0003181, -0.0268644, 0.0271804
5: -0.0009829, 0.0500200, 0.0012626, 0.0496467, -0.0506297, 0.0487574
6: -0.0410436, 0.0554346, -0.0386754, 0.0543882, -0.0954317, 0.0941101
7: -0.0207958, 0.0156302, -0.0190889, 0.0144272, -0.0352230, 0.0347191
8: 0.6852626, 0.9560105, 0.6882600, 0.9518959, -0.2175426, 0.2168024
9: 0.0432518, 0.0942075, 0.0455112, 0.0938498, -0.0505980, 0.0486963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.49 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0510145, 0.0641234, -0.0436411, 0.0592813, -0.1102958, 0.1077645
1: -0.0190375, 0.0191981, -0.0148572, 0.0144485, -0.0334861, 0.0340553
2: -0.0082251, 0.0386989, -0.0031229, 0.0370446, -0.0452697, 0.0418218
3: -0.0107923, 0.0705199, -0.0043976, 0.0673565, -0.0699743, 0.0668331
4: -0.0271825, 0.0008495, -0.0248410, -0.0023608, -0.0248217, 0.0256905
5: -0.0009828, 0.0497642, 0.0051905, 0.0480673, -0.0476800, 0.0445738
6: -0.0410434, 0.0547176, -0.0345330, 0.0499602, -0.0910036, 0.0892506
7: -0.0207957, 0.0156302, -0.0161031, 0.0123225, -0.0331183, 0.0317333
8: 0.6873170, 0.9560101, 0.7009428, 0.9446972, -0.2091103, 0.2065470
9: 0.0432518, 0.0939624, 0.0494637, 0.0923364, -0.0490846, 0.0444987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0510148, 0.0634804, -0.0478678, 0.0645912, -0.1156060, 0.1113483
1: -0.0190375, 0.0191981, -0.0172537, 0.0171715, -0.0362090, 0.0364518
2: -0.0082250, 0.0384792, -0.0060478, 0.0388587, -0.0470838, 0.0445270
3: -0.0107923, 0.0700997, -0.0080637, 0.0708254, -0.0729988, 0.0702797
4: -0.0271825, 0.0008495, -0.0261833, -0.0005204, -0.0266621, 0.0270329
5: -0.0009829, 0.0495388, 0.0016516, 0.0499281, -0.0506250, 0.0478873
6: -0.0410433, 0.0540856, -0.0382654, 0.0551771, -0.0962204, 0.0923510
7: -0.0207958, 0.0156302, -0.0187933, 0.0142187, -0.0350145, 0.0344236
8: 0.6891262, 0.9560096, 0.6860011, 0.9511826, -0.2151041, 0.2183177
9: 0.0432520, 0.0937465, 0.0459027, 0.0941195, -0.0508675, 0.0478438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1039127
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.49 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1086641
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0691003, -0.0483788, 0.0643713, -0.1122595, 0.1174790
1: -0.0172653, 0.0171843, -0.0175432, 0.0175002, -0.0347654, 0.0347275
2: -0.0060618, 0.0403992, -0.0064011, 0.0387837, -0.0448455, 0.0468003
3: -0.0080810, 0.0737708, -0.0085063, 0.0706818, -0.0708402, 0.0740180
4: -0.0261897, -0.0005116, -0.0263454, -0.0002981, -0.0258916, 0.0258338
5: 0.0016344, 0.0515081, 0.0012239, 0.0498511, -0.0482167, 0.0492357
6: -0.0382837, 0.0596068, -0.0387161, 0.0549611, -0.0932448, 0.0983229
7: -0.0188062, 0.0142279, -0.0191182, 0.0144478, -0.0332540, 0.0333461
8: 0.6733133, 0.9512133, 0.6866189, 0.9519659, -0.2291911, 0.2174633
9: 0.0458857, 0.0956335, 0.0454726, 0.0940456, -0.0481600, 0.0501609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0478881, 0.0697145, -0.0479112, 0.0654786, -0.1133667, 0.1176257
1: -0.0172652, 0.0171843, -0.0172781, 0.0171990, -0.0344642, 0.0344623
2: -0.0060618, 0.0406091, -0.0060775, 0.0391620, -0.0452238, 0.0466866
3: -0.0080812, 0.0741722, -0.0081012, 0.0714051, -0.0715061, 0.0743093
4: -0.0261897, -0.0005115, -0.0261970, -0.0005017, -0.0256880, 0.0256854
5: 0.0016343, 0.0517235, 0.0016154, 0.0502390, -0.0479315, 0.0491112
6: -0.0382832, 0.0602105, -0.0383031, 0.0560488, -0.0943320, 0.0985136
7: -0.0188062, 0.0142278, -0.0188207, 0.0142381, -0.0330443, 0.0330485
8: 0.6715845, 0.9512134, 0.6835042, 0.9512478, -0.2322598, 0.2198553
9: 0.0458857, 0.0958399, 0.0458664, 0.0944174, -0.0485317, 0.0499734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.51 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0483548, 0.0655823, -0.0436862, 0.0600004, -0.1083551, 0.1092684
1: -0.0175295, 0.0174848, -0.0148830, 0.0144780, -0.0320075, 0.0323679
2: -0.0063843, 0.0391974, -0.0031544, 0.0372902, -0.0436746, 0.0423517
3: -0.0084855, 0.0714727, -0.0044372, 0.0678261, -0.0684291, 0.0676475
4: -0.0263378, -0.0003085, -0.0248554, -0.0023410, -0.0239968, 0.0245469
5: 0.0012439, 0.0502754, 0.0051522, 0.0483193, -0.0451638, 0.0451232
6: -0.0386947, 0.0561508, -0.0345731, 0.0506667, -0.0893614, 0.0907239
7: -0.0191030, 0.0144369, -0.0161321, 0.0123429, -0.0314459, 0.0305691
8: 0.6832125, 0.9519305, 0.6989187, 0.9447667, -0.2121742, 0.2059553
9: 0.0454927, 0.0944522, 0.0494252, 0.0925778, -0.0470851, 0.0450270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
time: 0.51 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0478883, 0.0666994, -0.0436862, 0.0600004, -0.1078887, 0.1103855
1: -0.0172653, 0.0171843, -0.0148830, 0.0144780, -0.0317432, 0.0320674
2: -0.0060618, 0.0395790, -0.0031544, 0.0372902, -0.0433520, 0.0427333
3: -0.0080812, 0.0722025, -0.0044372, 0.0678261, -0.0683596, 0.0686610
4: -0.0261897, -0.0005116, -0.0248554, -0.0023410, -0.0238487, 0.0243438
5: 0.0016346, 0.0506668, 0.0051522, 0.0483193, -0.0449901, 0.0450849
6: -0.0382832, 0.0572480, -0.0345731, 0.0506667, -0.0889500, 0.0918211
7: -0.0188062, 0.0142278, -0.0161321, 0.0123429, -0.0311491, 0.0303600
8: 0.6800689, 0.9512141, 0.6989187, 0.9447667, -0.2173386, 0.2073939
9: 0.0458855, 0.0948273, 0.0494252, 0.0925778, -0.0466924, 0.0454021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
time: 0.51 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0478881, 0.0674547, -0.0483787, 0.0659988, -0.1138869, 0.1158335
1: -0.0172653, 0.0171843, -0.0175432, 0.0175002, -0.0347655, 0.0347274
2: -0.0060618, 0.0398371, -0.0064009, 0.0393397, -0.0454015, 0.0462380
3: -0.0080812, 0.0726959, -0.0085063, 0.0717448, -0.0719548, 0.0728996
4: -0.0261897, -0.0005115, -0.0263454, -0.0002981, -0.0258916, 0.0258339
5: 0.0016345, 0.0509315, 0.0012239, 0.0504213, -0.0487868, 0.0486155
6: -0.0382833, 0.0579901, -0.0387159, 0.0565598, -0.0948430, 0.0967060
7: -0.0188062, 0.0142278, -0.0191182, 0.0144477, -0.0332539, 0.0333461
8: 0.6779441, 0.9512135, 0.6820396, 0.9519657, -0.2243340, 0.2223315
9: 0.0458855, 0.0950809, 0.0454723, 0.0945921, -0.0487066, 0.0496086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0479110, 0.0671977, -0.1150859, 0.1159527
1: -0.0172653, 0.0171843, -0.0172780, 0.0171990, -0.0344643, 0.0344623
2: -0.0060618, 0.0400376, -0.0060776, 0.0397492, -0.0458111, 0.0461152
3: -0.0080811, 0.0730793, -0.0081008, 0.0725281, -0.0726549, 0.0731865
4: -0.0261897, -0.0005116, -0.0261969, -0.0005017, -0.0256880, 0.0256853
5: 0.0016343, 0.0511372, 0.0016153, 0.0508415, -0.0485761, 0.0484890
6: -0.0382830, 0.0585671, -0.0383031, 0.0577376, -0.0960205, 0.0968701
7: -0.0188062, 0.0142279, -0.0188207, 0.0142381, -0.0330442, 0.0330486
8: 0.6762923, 0.9512142, 0.6786667, 0.9512488, -0.2273870, 0.2248356
9: 0.0458855, 0.0952781, 0.0458664, 0.0949947, -0.0491092, 0.0494117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483787, 0.0659988, -0.1093056, 0.1115580
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064009, 0.0393397, -0.0422318, 0.0447773
3: -0.0041086, 0.0699030, -0.0085063, 0.0717448, -0.0680323, 0.0703536
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238394
5: 0.0054695, 0.0494333, 0.0012239, 0.0504213, -0.0449518, 0.0453321
6: -0.0342385, 0.0537900, -0.0387159, 0.0565598, -0.0907982, 0.0925059
7: -0.0158908, 0.0121729, -0.0191182, 0.0144477, -0.0303385, 0.0312911
8: 0.6899735, 0.9441853, 0.6820396, 0.9519657, -0.2137654, 0.2157047
9: 0.0497447, 0.0936454, 0.0454723, 0.0945921, -0.0448474, 0.0481731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479110, 0.0671977, -0.1105049, 0.1116859
1: -0.0146681, 0.0142336, -0.0172780, 0.0171990, -0.0318671, 0.0315116
2: -0.0028921, 0.0385798, -0.0060776, 0.0397492, -0.0426413, 0.0446574
3: -0.0041084, 0.0702920, -0.0081008, 0.0725281, -0.0687342, 0.0706360
4: -0.0247350, -0.0024921, -0.0261969, -0.0005017, -0.0242333, 0.0237048
5: 0.0054697, 0.0496420, 0.0016153, 0.0508415, -0.0447752, 0.0452269
6: -0.0342384, 0.0543750, -0.0383031, 0.0577376, -0.0919759, 0.0926780
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301288, 0.0309935
8: 0.6882984, 0.9441848, 0.6786667, 0.9512488, -0.2168388, 0.2182202
9: 0.0497448, 0.0938453, 0.0458664, 0.0949947, -0.0452499, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167
time: 0.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.37 seconds
IS_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1039127
IS_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167

## BFS IS instance: IS_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0510147, 0.0647588, -0.0483328, 0.0637884, -0.1148030, 0.1130916
1: -0.0190376, 0.0191981, -0.0175171, 0.0174706, -0.0365082, 0.0367152
2: -0.0082249, 0.0389161, -0.0063692, 0.0385845, -0.0468094, 0.0452852
3: -0.0107923, 0.0709348, -0.0084666, 0.0703007, -0.0725117, 0.0711887
4: -0.0271825, 0.0008495, -0.0263309, -0.0003181, -0.0268643, 0.0271804
5: -0.0009829, 0.0499868, 0.0012626, 0.0496467, -0.0506296, 0.0487242
6: -0.0410434, 0.0553416, -0.0386754, 0.0543882, -0.0954316, 0.0940170
7: -0.0207958, 0.0156302, -0.0190889, 0.0144272, -0.0352230, 0.0347191
8: 0.6855294, 0.9560105, 0.6882600, 0.9518959, -0.2172661, 0.2168026
9: 0.0432519, 0.0941757, 0.0455112, 0.0938498, -0.0505979, 0.0486645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0510144, 0.0648657, -0.0478678, 0.0645912, -0.1156056, 0.1127336
1: -0.0190375, 0.0191981, -0.0172537, 0.0171715, -0.0362090, 0.0364518
2: -0.0082250, 0.0389526, -0.0060478, 0.0388587, -0.0470838, 0.0450003
3: -0.0107923, 0.0710048, -0.0080637, 0.0708254, -0.0729987, 0.0711828
4: -0.0271825, 0.0008495, -0.0261833, -0.0005204, -0.0266621, 0.0270329
5: -0.0009829, 0.0500243, 0.0016516, 0.0499281, -0.0506250, 0.0483727
6: -0.0410436, 0.0554467, -0.0382654, 0.0551771, -0.0962207, 0.0937121
7: -0.0207957, 0.0156302, -0.0187933, 0.0142187, -0.0350144, 0.0344235
8: 0.6852283, 0.9560106, 0.6860011, 0.9511826, -0.2189865, 0.2183182
9: 0.0432520, 0.0942116, 0.0459027, 0.0941195, -0.0508675, 0.0483089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0510147, 0.0647588, -0.0436411, 0.0592813, -0.1102960, 0.1083999
1: -0.0190376, 0.0191981, -0.0148572, 0.0144485, -0.0334861, 0.0340553
2: -0.0082249, 0.0389161, -0.0031229, 0.0370446, -0.0452695, 0.0420389
3: -0.0107923, 0.0709348, -0.0043976, 0.0673565, -0.0699745, 0.0672492
4: -0.0271825, 0.0008495, -0.0248410, -0.0023608, -0.0248216, 0.0256906
5: -0.0009829, 0.0499868, 0.0051905, 0.0480673, -0.0476803, 0.0447963
6: -0.0410434, 0.0553416, -0.0345330, 0.0499602, -0.0910036, 0.0898746
7: -0.0207958, 0.0156302, -0.0161031, 0.0123225, -0.0331183, 0.0317333
8: 0.6855294, 0.9560105, 0.7009428, 0.9446972, -0.2109051, 0.2065475
9: 0.0432519, 0.0941757, 0.0494637, 0.0923364, -0.0490845, 0.0447120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0510144, 0.0648657, -0.0432869, 0.0601637, -0.1111781, 0.1081526
1: -0.0190375, 0.0191981, -0.0146567, 0.0142207, -0.0332583, 0.0338549
2: -0.0082250, 0.0389526, -0.0028783, 0.0373461, -0.0455711, 0.0418309
3: -0.0107923, 0.0710048, -0.0040907, 0.0679330, -0.0705031, 0.0672598
4: -0.0271825, 0.0008495, -0.0247287, -0.0025147, -0.0246677, 0.0255782
5: -0.0009829, 0.0500243, 0.0054865, 0.0483766, -0.0473337, 0.0445378
6: -0.0410436, 0.0554467, -0.0342208, 0.0508273, -0.0918709, 0.0896675
7: -0.0207957, 0.0156302, -0.0158781, 0.0121639, -0.0329596, 0.0315083
8: 0.6852283, 0.9560106, 0.6984591, 0.9441545, -0.2123556, 0.2082403
9: 0.0432520, 0.0942116, 0.0497617, 0.0926327, -0.0493808, 0.0444500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0483328, 0.0637884, -0.1101855, 0.1066271
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337410
2: -0.0050301, 0.0367074, -0.0063692, 0.0385845, -0.0436146, 0.0430765
3: -0.0067880, 0.0667118, -0.0084666, 0.0703007, -0.0685304, 0.0669785
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028829, 0.0477215, 0.0012626, 0.0496467, -0.0467639, 0.0464589
6: -0.0369667, 0.0489906, -0.0386754, 0.0543882, -0.0913549, 0.0876660
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326478
8: 0.7037196, 0.9489262, 0.6882600, 0.9518959, -0.1992197, 0.2102108
9: 0.0471417, 0.0920050, 0.0455112, 0.0938498, -0.0467081, 0.0464938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0436411, 0.0592813, -0.1056784, 0.1019353
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050301, 0.0367074, -0.0031229, 0.0370446, -0.0420747, 0.0398302
3: -0.0067880, 0.0667118, -0.0043976, 0.0673565, -0.0663639, 0.0637336
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028829, 0.0477215, 0.0051905, 0.0480673, -0.0440732, 0.0425310
6: -0.0369667, 0.0489906, -0.0345330, 0.0499602, -0.0869269, 0.0835236
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301797, 0.0296620
8: 0.7037196, 0.9489262, 0.7009428, 0.9446972, -0.1967988, 0.2020478
9: 0.0471417, 0.0920050, 0.0494637, 0.0923364, -0.0451947, 0.0425412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1026813
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0483788, 0.0643713, -0.1122595, 0.1164205
1: -0.0172653, 0.0171843, -0.0175432, 0.0175002, -0.0347654, 0.0347275
2: -0.0060618, 0.0400376, -0.0064011, 0.0387837, -0.0448455, 0.0464387
3: -0.0080811, 0.0730793, -0.0085063, 0.0706818, -0.0708404, 0.0733056
4: -0.0261897, -0.0005116, -0.0263454, -0.0002981, -0.0258916, 0.0258338
5: 0.0016343, 0.0511372, 0.0012239, 0.0498511, -0.0482168, 0.0486927
6: -0.0382830, 0.0585671, -0.0387161, 0.0549611, -0.0932441, 0.0972831
7: -0.0188062, 0.0142279, -0.0191182, 0.0144478, -0.0332540, 0.0333461
8: 0.6762923, 0.9512142, 0.6866189, 0.9519659, -0.2261345, 0.2174623
9: 0.0458855, 0.0952781, 0.0454726, 0.0940456, -0.0481602, 0.0498055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0483788, 0.0643713, -0.1076785, 0.1121537
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0385798, -0.0064011, 0.0387837, -0.0416758, 0.0449809
3: -0.0041084, 0.0702920, -0.0085063, 0.0706818, -0.0669177, 0.0705454
4: -0.0247350, -0.0024921, -0.0263454, -0.0002981, -0.0244369, 0.0238533
5: 0.0054697, 0.0496420, 0.0012239, 0.0498511, -0.0443814, 0.0461946
6: -0.0342384, 0.0543750, -0.0387161, 0.0549611, -0.0891995, 0.0930911
7: -0.0158908, 0.0121728, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6882984, 0.9441848, 0.6866189, 0.9519659, -0.2143590, 0.2108366
9: 0.0497448, 0.0938453, 0.0454726, 0.0940456, -0.0443008, 0.0483727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.51 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0479112, 0.0654786, -0.1133668, 0.1159529
1: -0.0172653, 0.0171843, -0.0172781, 0.0171990, -0.0344642, 0.0344624
2: -0.0060618, 0.0400376, -0.0060775, 0.0391620, -0.0452238, 0.0461151
3: -0.0080811, 0.0730793, -0.0081012, 0.0714051, -0.0715061, 0.0731862
4: -0.0261897, -0.0005116, -0.0261970, -0.0005017, -0.0256880, 0.0256854
5: 0.0016343, 0.0511372, 0.0016154, 0.0502390, -0.0479315, 0.0484888
6: -0.0382830, 0.0585671, -0.0383031, 0.0560488, -0.0943318, 0.0968702
7: -0.0188062, 0.0142279, -0.0188207, 0.0142381, -0.0330443, 0.0330486
8: 0.6762923, 0.9512142, 0.6835042, 0.9512478, -0.2273865, 0.2198558
9: 0.0458855, 0.0952781, 0.0458664, 0.0944174, -0.0485320, 0.0494117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0436862, 0.0600004, -0.1036627, 0.1032854
1: -0.0148696, 0.0144624, -0.0148830, 0.0144780, -0.0293476, 0.0293454
2: -0.0031379, 0.0371533, -0.0031544, 0.0372902, -0.0404281, 0.0403077
3: -0.0044166, 0.0675643, -0.0044372, 0.0678261, -0.0648879, 0.0643740
4: -0.0248479, -0.0023514, -0.0248554, -0.0023410, -0.0225069, 0.0225040
5: 0.0051722, 0.0481788, 0.0051522, 0.0483193, -0.0415780, 0.0415182
6: -0.0345522, 0.0502727, -0.0345731, 0.0506667, -0.0852190, 0.0848458
7: -0.0161169, 0.0123323, -0.0161321, 0.0123429, -0.0284598, 0.0284644
8: 0.7000468, 0.9447302, 0.6989187, 0.9447667, -0.1989958, 0.2017820
9: 0.0494456, 0.0924433, 0.0494252, 0.0925778, -0.0431323, 0.0430180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144414, upper bound: 0.1144016
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144414, upper bound: 0.1144016
time: 0.52 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0436862, 0.0600004, -0.1033072, 0.1043972
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028922, 0.0375331, -0.0031544, 0.0372902, -0.0401824, 0.0406874
3: -0.0041084, 0.0682905, -0.0044372, 0.0678261, -0.0648521, 0.0653646
4: -0.0247351, -0.0025060, -0.0248554, -0.0023410, -0.0223941, 0.0223494
5: 0.0054697, 0.0485683, 0.0051522, 0.0483193, -0.0414588, 0.0414622
6: -0.0342384, 0.0513647, -0.0345731, 0.0506667, -0.0849051, 0.0859378
7: -0.0158908, 0.0121729, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6969191, 0.9441855, 0.6989187, 0.9447667, -0.2041392, 0.2029879
9: 0.0497447, 0.0928164, 0.0494252, 0.0925778, -0.0428331, 0.0433912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144016
time: 0.51 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0483788, 0.0643713, -0.1122595, 0.1164205
1: -0.0172653, 0.0171843, -0.0175432, 0.0175002, -0.0347654, 0.0347275
2: -0.0060618, 0.0400376, -0.0064011, 0.0387837, -0.0448455, 0.0464387
3: -0.0080811, 0.0730793, -0.0085063, 0.0706818, -0.0708404, 0.0733056
4: -0.0261897, -0.0005116, -0.0263454, -0.0002981, -0.0258916, 0.0258338
5: 0.0016343, 0.0511372, 0.0012239, 0.0498511, -0.0482168, 0.0486927
6: -0.0382830, 0.0585671, -0.0387161, 0.0549611, -0.0932441, 0.0972831
7: -0.0188062, 0.0142279, -0.0191182, 0.0144478, -0.0332540, 0.0333461
8: 0.6762923, 0.9512142, 0.6866189, 0.9519659, -0.2261345, 0.2174623
9: 0.0458855, 0.0952781, 0.0454726, 0.0940456, -0.0481602, 0.0498055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0436862, 0.0600004, -0.1078885, 0.1117279
1: -0.0172653, 0.0171843, -0.0148830, 0.0144780, -0.0317432, 0.0320673
2: -0.0060618, 0.0400376, -0.0031544, 0.0372902, -0.0433521, 0.0431920
3: -0.0080811, 0.0730793, -0.0044372, 0.0678261, -0.0680933, 0.0693383
4: -0.0261897, -0.0005116, -0.0248554, -0.0023410, -0.0238487, 0.0243439
5: 0.0016343, 0.0511372, 0.0051522, 0.0483193, -0.0457268, 0.0448328
6: -0.0382830, 0.0585671, -0.0345731, 0.0506667, -0.0889497, 0.0931402
7: -0.0188062, 0.0142279, -0.0161321, 0.0123429, -0.0311491, 0.0303600
8: 0.6762923, 0.9512142, 0.6989187, 0.9447667, -0.2196519, 0.2058556
9: 0.0458855, 0.0952781, 0.0494252, 0.0925778, -0.0466924, 0.0458529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0479110, 0.0671977, -0.1150859, 0.1159527
1: -0.0172653, 0.0171843, -0.0172780, 0.0171990, -0.0344643, 0.0344623
2: -0.0060618, 0.0400376, -0.0060776, 0.0397492, -0.0458111, 0.0461152
3: -0.0080811, 0.0730793, -0.0081008, 0.0725281, -0.0726549, 0.0731865
4: -0.0261897, -0.0005116, -0.0261969, -0.0005017, -0.0256880, 0.0256853
5: 0.0016343, 0.0511372, 0.0016153, 0.0508415, -0.0485761, 0.0484890
6: -0.0382830, 0.0585671, -0.0383031, 0.0577376, -0.0960205, 0.0968701
7: -0.0188062, 0.0142279, -0.0188207, 0.0142381, -0.0330442, 0.0330486
8: 0.6762923, 0.9512142, 0.6786667, 0.9512488, -0.2273870, 0.2248356
9: 0.0458855, 0.0952781, 0.0458664, 0.0949947, -0.0491092, 0.0494117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479108, 0.0662074, -0.1095146, 0.1116857
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0394109, -0.0423031, 0.0446573
3: -0.0041084, 0.0702920, -0.0081008, 0.0718812, -0.0680928, 0.0706359
4: -0.0247350, -0.0024921, -0.0261969, -0.0005017, -0.0242333, 0.0237048
5: 0.0054697, 0.0496420, 0.0016155, 0.0504945, -0.0447388, 0.0452268
6: -0.0342384, 0.0543750, -0.0383035, 0.0567648, -0.0910032, 0.0926785
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6814528, 0.9512495, -0.2168379, 0.2154725
9: 0.0497448, 0.0938453, 0.0458663, 0.0946622, -0.0449174, 0.0479790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1047672
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0483788, 0.0643713, -0.1076785, 0.1121537
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0385798, -0.0064011, 0.0387837, -0.0416758, 0.0449809
3: -0.0041084, 0.0702920, -0.0085063, 0.0706818, -0.0669177, 0.0705454
4: -0.0247350, -0.0024921, -0.0263454, -0.0002981, -0.0244369, 0.0238533
5: 0.0054697, 0.0496420, 0.0012239, 0.0498511, -0.0443814, 0.0461946
6: -0.0342384, 0.0543750, -0.0387161, 0.0549611, -0.0891995, 0.0930911
7: -0.0158908, 0.0121728, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6882984, 0.9441848, 0.6866189, 0.9519659, -0.2143590, 0.2108366
9: 0.0497448, 0.0938453, 0.0454726, 0.0940456, -0.0443008, 0.0483727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0436862, 0.0600004, -0.1033075, 0.1074611
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0385798, -0.0031544, 0.0372902, -0.0401823, 0.0417342
3: -0.0041084, 0.0702920, -0.0044372, 0.0678261, -0.0648525, 0.0671834
4: -0.0247350, -0.0024921, -0.0248554, -0.0023410, -0.0223940, 0.0223633
5: 0.0054697, 0.0496420, 0.0051522, 0.0483193, -0.0414594, 0.0418269
6: -0.0342384, 0.0543750, -0.0345731, 0.0506667, -0.0849051, 0.0889481
7: -0.0158908, 0.0121728, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6882984, 0.9441848, 0.6989187, 0.9447667, -0.2113264, 0.2029898
9: 0.0497448, 0.0938453, 0.0494252, 0.0925778, -0.0428330, 0.0444201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0433297, 0.0612006, -0.1045078, 0.1071046
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289146
2: -0.0028921, 0.0385798, -0.0029075, 0.0377003, -0.0405924, 0.0414873
3: -0.0041084, 0.0702920, -0.0041280, 0.0686103, -0.0655686, 0.0671036
4: -0.0247350, -0.0024921, -0.0247422, -0.0024963, -0.0222387, 0.0222501
5: 0.0054697, 0.0496420, 0.0054508, 0.0487399, -0.0411208, 0.0416791
6: -0.0342384, 0.0543750, -0.0342584, 0.0518458, -0.0860841, 0.0886334
7: -0.0158908, 0.0121728, -0.0159051, 0.0121830, -0.0280738, 0.0280780
8: 0.6882984, 0.9441848, 0.6955412, 0.9442201, -0.2123580, 0.2057641
9: 0.0497448, 0.0938453, 0.0497257, 0.0929809, -0.0432361, 0.0441196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144026
time: 0.53 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.42 seconds
IS_A1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1026813
IS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A2_B1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
IS_A2_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
IS_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1144414, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1144414, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1047672
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
IS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144026

## BFS IS instance: IS_A1_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0510148, 0.0648533, -0.0483328, 0.0637884, -0.1148032, 0.1131861
1: -0.0190376, 0.0191981, -0.0175171, 0.0174706, -0.0365082, 0.0367152
2: -0.0082249, 0.0389483, -0.0063692, 0.0385845, -0.0468094, 0.0453174
3: -0.0107924, 0.0709966, -0.0084666, 0.0703007, -0.0725119, 0.0712514
4: -0.0271825, 0.0008495, -0.0263309, -0.0003181, -0.0268644, 0.0271804
5: -0.0009829, 0.0500200, 0.0012626, 0.0496467, -0.0506297, 0.0487574
6: -0.0410436, 0.0554346, -0.0386754, 0.0543882, -0.0954317, 0.0941101
7: -0.0207958, 0.0156302, -0.0190889, 0.0144272, -0.0352230, 0.0347191
8: 0.6852626, 0.9560105, 0.6882600, 0.9518959, -0.2175426, 0.2168024
9: 0.0432518, 0.0942075, 0.0455112, 0.0938498, -0.0505980, 0.0486963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0510145, 0.0641234, -0.0436411, 0.0592813, -0.1102958, 0.1077645
1: -0.0190375, 0.0191981, -0.0148572, 0.0144485, -0.0334861, 0.0340553
2: -0.0082251, 0.0386989, -0.0031229, 0.0370446, -0.0452697, 0.0418218
3: -0.0107923, 0.0705199, -0.0043976, 0.0673565, -0.0699743, 0.0668331
4: -0.0271825, 0.0008495, -0.0248410, -0.0023608, -0.0248217, 0.0256905
5: -0.0009828, 0.0497642, 0.0051905, 0.0480673, -0.0476800, 0.0445738
6: -0.0410434, 0.0547176, -0.0345330, 0.0499602, -0.0910036, 0.0892506
7: -0.0207957, 0.0156302, -0.0161031, 0.0123225, -0.0331183, 0.0317333
8: 0.6873170, 0.9560101, 0.7009428, 0.9446972, -0.2091103, 0.2065470
9: 0.0432518, 0.0939624, 0.0494637, 0.0923364, -0.0490846, 0.0444987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0510148, 0.0634804, -0.0478678, 0.0645912, -0.1156060, 0.1113483
1: -0.0190375, 0.0191981, -0.0172537, 0.0171715, -0.0362090, 0.0364518
2: -0.0082250, 0.0384792, -0.0060478, 0.0388587, -0.0470838, 0.0445270
3: -0.0107923, 0.0700997, -0.0080637, 0.0708254, -0.0729988, 0.0702797
4: -0.0271825, 0.0008495, -0.0261833, -0.0005204, -0.0266621, 0.0270329
5: -0.0009829, 0.0495388, 0.0016516, 0.0499281, -0.0506250, 0.0478873
6: -0.0410433, 0.0540856, -0.0382654, 0.0551771, -0.0962204, 0.0923510
7: -0.0207958, 0.0156302, -0.0187933, 0.0142187, -0.0350145, 0.0344236
8: 0.6891262, 0.9560096, 0.6860011, 0.9511826, -0.2151041, 0.2183177
9: 0.0432520, 0.0937465, 0.0459027, 0.0941195, -0.0508675, 0.0478438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.65 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.61 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.49 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0478678, 0.0645912, -0.1110106, 0.1033163
1: -0.0164326, 0.0162384, -0.0172537, 0.0171715, -0.0336040, 0.0334921
2: -0.0050456, 0.0357351, -0.0060478, 0.0388587, -0.0439044, 0.0417828
3: -0.0068073, 0.0648526, -0.0080637, 0.0708254, -0.0692218, 0.0650368
4: -0.0257235, -0.0011510, -0.0261833, -0.0005204, -0.0252031, 0.0250324
5: 0.0028639, 0.0467242, 0.0016516, 0.0499281, -0.0469098, 0.0450726
6: -0.0369866, 0.0461946, -0.0382654, 0.0551771, -0.0921637, 0.0844600
7: -0.0178716, 0.0135691, -0.0187933, 0.0142187, -0.0320902, 0.0323624
8: 0.7117278, 0.9489607, 0.6860011, 0.9511826, -0.1925888, 0.2129822
9: 0.0471227, 0.0910493, 0.0459027, 0.0941195, -0.0469968, 0.0451466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1086967
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1086967
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0478678, 0.0645912, -0.1109878, 0.1024290
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334776
2: -0.0050301, 0.0354300, -0.0060478, 0.0388587, -0.0438888, 0.0414778
3: -0.0067881, 0.0642716, -0.0080637, 0.0708254, -0.0690598, 0.0643206
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0464113, 0.0016516, 0.0499281, -0.0468019, 0.0447597
6: -0.0369667, 0.0453175, -0.0382654, 0.0551771, -0.0921438, 0.0835829
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320758, 0.0323523
8: 0.7142399, 0.9489262, 0.6860011, 0.9511826, -0.1888852, 0.2116635
9: 0.0471418, 0.0907734, 0.0459027, 0.0941195, -0.0469777, 0.0448707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483788, 0.0643713, -0.1076782, 0.1115581
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064011, 0.0387837, -0.0416758, 0.0447775
3: -0.0041086, 0.0699030, -0.0085063, 0.0706818, -0.0669177, 0.0701455
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054695, 0.0494333, 0.0012239, 0.0498511, -0.0443815, 0.0460978
6: -0.0342385, 0.0537900, -0.0387161, 0.0549611, -0.0891996, 0.0925061
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6899735, 0.9441853, 0.6866189, 0.9519659, -0.2126038, 0.2108355
9: 0.0497447, 0.0936454, 0.0454726, 0.0940456, -0.0443009, 0.0481727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483788, 0.0643713, -0.1076782, 0.1115581
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064011, 0.0387837, -0.0416758, 0.0447775
3: -0.0041086, 0.0699030, -0.0085063, 0.0706818, -0.0669177, 0.0701455
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054695, 0.0494333, 0.0012239, 0.0498511, -0.0443815, 0.0460978
6: -0.0342385, 0.0537900, -0.0387161, 0.0549611, -0.0891996, 0.0925061
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6899735, 0.9441853, 0.6866189, 0.9519659, -0.2126038, 0.2108355
9: 0.0497447, 0.0936454, 0.0454726, 0.0940456, -0.0443009, 0.0481727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0436862, 0.0600004, -0.1036627, 0.1032854
1: -0.0148696, 0.0144624, -0.0148830, 0.0144780, -0.0293476, 0.0293454
2: -0.0031379, 0.0371533, -0.0031544, 0.0372902, -0.0404281, 0.0403077
3: -0.0044166, 0.0675643, -0.0044372, 0.0678261, -0.0648879, 0.0643740
4: -0.0248479, -0.0023514, -0.0248554, -0.0023410, -0.0225069, 0.0225040
5: 0.0051722, 0.0481788, 0.0051522, 0.0483193, -0.0415780, 0.0415182
6: -0.0345522, 0.0502727, -0.0345731, 0.0506667, -0.0852190, 0.0848458
7: -0.0161169, 0.0123323, -0.0161321, 0.0123429, -0.0284598, 0.0284644
8: 0.7000468, 0.9447302, 0.6989187, 0.9447667, -0.1989958, 0.2017820
9: 0.0494456, 0.0924433, 0.0494252, 0.0925778, -0.0431323, 0.0430180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0433297, 0.0612006, -0.1048630, 0.1029289
1: -0.0148696, 0.0144624, -0.0146810, 0.0142479, -0.0291175, 0.0291434
2: -0.0031379, 0.0371533, -0.0029075, 0.0377003, -0.0408382, 0.0400608
3: -0.0044166, 0.0675643, -0.0041280, 0.0686103, -0.0659669, 0.0642941
4: -0.0248479, -0.0023514, -0.0247422, -0.0024963, -0.0223516, 0.0223908
5: 0.0051722, 0.0481788, 0.0054508, 0.0487399, -0.0416168, 0.0413705
6: -0.0345522, 0.0502727, -0.0342584, 0.0518458, -0.0863980, 0.0845311
7: -0.0161169, 0.0123323, -0.0159051, 0.0121830, -0.0282999, 0.0282374
8: 0.7000468, 0.9447302, 0.6955412, 0.9442201, -0.2000279, 0.2073538
9: 0.0494456, 0.0924433, 0.0497257, 0.0929809, -0.0435354, 0.0427175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0436862, 0.0600004, -0.1033072, 0.1043972
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028922, 0.0375331, -0.0031544, 0.0372902, -0.0401824, 0.0406874
3: -0.0041084, 0.0682905, -0.0044372, 0.0678261, -0.0648521, 0.0653646
4: -0.0247351, -0.0025060, -0.0248554, -0.0023410, -0.0223941, 0.0223494
5: 0.0054697, 0.0485683, 0.0051522, 0.0483193, -0.0414588, 0.0414622
6: -0.0342384, 0.0513647, -0.0345731, 0.0506667, -0.0849051, 0.0859378
7: -0.0158908, 0.0121729, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6969191, 0.9441855, 0.6989187, 0.9447667, -0.2041392, 0.2029879
9: 0.0497447, 0.0928164, 0.0494252, 0.0925778, -0.0428331, 0.0433912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0433297, 0.0612006, -0.1045074, 0.1040407
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289145
2: -0.0028922, 0.0375331, -0.0029075, 0.0377003, -0.0405925, 0.0404406
3: -0.0041084, 0.0682905, -0.0041280, 0.0686103, -0.0655687, 0.0647458
4: -0.0247351, -0.0025060, -0.0247422, -0.0024963, -0.0222388, 0.0222361
5: 0.0054697, 0.0485683, 0.0054508, 0.0487399, -0.0411207, 0.0410181
6: -0.0342384, 0.0513647, -0.0342584, 0.0518458, -0.0860842, 0.0856232
7: -0.0158908, 0.0121729, -0.0159051, 0.0121830, -0.0280738, 0.0280781
8: 0.6969191, 0.9441855, 0.6955412, 0.9442201, -0.2012339, 0.2057638
9: 0.0497447, 0.0928164, 0.0497257, 0.0929809, -0.0432362, 0.0430907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0478881, 0.0674547, -0.0483787, 0.0659988, -0.1138869, 0.1158335
1: -0.0172653, 0.0171843, -0.0175432, 0.0175002, -0.0347655, 0.0347274
2: -0.0060618, 0.0398371, -0.0064009, 0.0393397, -0.0454015, 0.0462380
3: -0.0080812, 0.0726959, -0.0085063, 0.0717448, -0.0719548, 0.0728996
4: -0.0261897, -0.0005115, -0.0263454, -0.0002981, -0.0258916, 0.0258339
5: 0.0016345, 0.0509315, 0.0012239, 0.0504213, -0.0487868, 0.0486155
6: -0.0382833, 0.0579901, -0.0387159, 0.0565598, -0.0948430, 0.0967060
7: -0.0188062, 0.0142278, -0.0191182, 0.0144477, -0.0332539, 0.0333461
8: 0.6779441, 0.9512135, 0.6820396, 0.9519657, -0.2243340, 0.2223315
9: 0.0458855, 0.0950809, 0.0454723, 0.0945921, -0.0487066, 0.0496086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0479110, 0.0671977, -0.1150859, 0.1159527
1: -0.0172653, 0.0171843, -0.0172780, 0.0171990, -0.0344643, 0.0344623
2: -0.0060618, 0.0400376, -0.0060776, 0.0397492, -0.0458111, 0.0461152
3: -0.0080811, 0.0730793, -0.0081008, 0.0725281, -0.0726549, 0.0731865
4: -0.0261897, -0.0005116, -0.0261969, -0.0005017, -0.0256880, 0.0256853
5: 0.0016343, 0.0511372, 0.0016153, 0.0508415, -0.0485761, 0.0484890
6: -0.0382830, 0.0585671, -0.0383031, 0.0577376, -0.0960205, 0.0968701
7: -0.0188062, 0.0142279, -0.0188207, 0.0142381, -0.0330442, 0.0330486
8: 0.6762923, 0.9512142, 0.6786667, 0.9512488, -0.2273870, 0.2248356
9: 0.0458855, 0.0952781, 0.0458664, 0.0949947, -0.0491092, 0.0494117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483787, 0.0659988, -0.1093056, 0.1115580
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064009, 0.0393397, -0.0422318, 0.0447773
3: -0.0041086, 0.0699030, -0.0085063, 0.0717448, -0.0680323, 0.0703536
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238394
5: 0.0054695, 0.0494333, 0.0012239, 0.0504213, -0.0449518, 0.0453321
6: -0.0342385, 0.0537900, -0.0387159, 0.0565598, -0.0907982, 0.0925059
7: -0.0158908, 0.0121729, -0.0191182, 0.0144477, -0.0303385, 0.0312911
8: 0.6899735, 0.9441853, 0.6820396, 0.9519657, -0.2137654, 0.2157047
9: 0.0497447, 0.0936454, 0.0454723, 0.0945921, -0.0448474, 0.0481731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479110, 0.0671977, -0.1105049, 0.1116859
1: -0.0146681, 0.0142336, -0.0172780, 0.0171990, -0.0318671, 0.0315116
2: -0.0028921, 0.0385798, -0.0060776, 0.0397492, -0.0426413, 0.0446574
3: -0.0041084, 0.0702920, -0.0081008, 0.0725281, -0.0687342, 0.0706360
4: -0.0247350, -0.0024921, -0.0261969, -0.0005017, -0.0242333, 0.0237048
5: 0.0054697, 0.0496420, 0.0016153, 0.0508415, -0.0447752, 0.0452269
6: -0.0342384, 0.0543750, -0.0383031, 0.0577376, -0.0919759, 0.0926780
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301288, 0.0309935
8: 0.6882984, 0.9441848, 0.6786667, 0.9512488, -0.2168388, 0.2182202
9: 0.0497448, 0.0938453, 0.0458664, 0.0949947, -0.0452499, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1057859
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1057859
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483788, 0.0643713, -0.1076782, 0.1115581
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064011, 0.0387837, -0.0416758, 0.0447775
3: -0.0041086, 0.0699030, -0.0085063, 0.0706818, -0.0669177, 0.0701455
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054695, 0.0494333, 0.0012239, 0.0498511, -0.0443815, 0.0460978
6: -0.0342385, 0.0537900, -0.0387161, 0.0549611, -0.0891996, 0.0925061
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6899735, 0.9441853, 0.6866189, 0.9519659, -0.2126038, 0.2108355
9: 0.0497447, 0.0936454, 0.0454726, 0.0940456, -0.0443009, 0.0481727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0436862, 0.0600004, -0.1033072, 0.1068655
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0383764, -0.0031544, 0.0372902, -0.0401824, 0.0415308
3: -0.0041086, 0.0699030, -0.0044372, 0.0678261, -0.0648524, 0.0667820
4: -0.0247351, -0.0025060, -0.0248554, -0.0023410, -0.0223941, 0.0223494
5: 0.0054695, 0.0494333, 0.0051522, 0.0483193, -0.0414594, 0.0417284
6: -0.0342385, 0.0537900, -0.0345731, 0.0506667, -0.0849052, 0.0883631
7: -0.0158908, 0.0121729, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6899735, 0.9441853, 0.6989187, 0.9447667, -0.2095048, 0.2029891
9: 0.0497447, 0.0936454, 0.0494252, 0.0925778, -0.0428331, 0.0442201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144414
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0433297, 0.0612006, -0.1045078, 0.1071046
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289146
2: -0.0028921, 0.0385798, -0.0029075, 0.0377003, -0.0405924, 0.0414873
3: -0.0041084, 0.0702920, -0.0041280, 0.0686103, -0.0655686, 0.0671036
4: -0.0247350, -0.0024921, -0.0247422, -0.0024963, -0.0222387, 0.0222501
5: 0.0054697, 0.0496420, 0.0054508, 0.0487399, -0.0411208, 0.0416791
6: -0.0342384, 0.0543750, -0.0342584, 0.0518458, -0.0860841, 0.0886334
7: -0.0158908, 0.0121728, -0.0159051, 0.0121830, -0.0280738, 0.0280780
8: 0.6882984, 0.9441848, 0.6955412, 0.9442201, -0.2123580, 0.2057641
9: 0.0497448, 0.0938453, 0.0497257, 0.0929809, -0.0432361, 0.0441196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483788, 0.0643713, -0.1076782, 0.1115581
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064011, 0.0387837, -0.0416758, 0.0447775
3: -0.0041086, 0.0699030, -0.0085063, 0.0706818, -0.0669177, 0.0701455
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054695, 0.0494333, 0.0012239, 0.0498511, -0.0443815, 0.0460978
6: -0.0342385, 0.0537900, -0.0387161, 0.0549611, -0.0891996, 0.0925061
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6899735, 0.9441853, 0.6866189, 0.9519659, -0.2126038, 0.2108355
9: 0.0497447, 0.0936454, 0.0454726, 0.0940456, -0.0443009, 0.0481727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1218361
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1218361
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0433297, 0.0612006, -0.1048630, 0.1029289
1: -0.0148696, 0.0144624, -0.0146810, 0.0142479, -0.0291175, 0.0291434
2: -0.0031379, 0.0371533, -0.0029075, 0.0377003, -0.0408382, 0.0400608
3: -0.0044166, 0.0675643, -0.0041280, 0.0686103, -0.0659669, 0.0642941
4: -0.0248479, -0.0023514, -0.0247422, -0.0024963, -0.0223516, 0.0223908
5: 0.0051722, 0.0481788, 0.0054508, 0.0487399, -0.0416168, 0.0413705
6: -0.0345522, 0.0502727, -0.0342584, 0.0518458, -0.0863980, 0.0845311
7: -0.0161169, 0.0123323, -0.0159051, 0.0121830, -0.0282999, 0.0282374
8: 0.7000468, 0.9447302, 0.6955412, 0.9442201, -0.2000279, 0.2073538
9: 0.0494456, 0.0924433, 0.0497257, 0.0929809, -0.0435354, 0.0427175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1149167
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0433297, 0.0612006, -0.1045074, 0.1040407
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289145
2: -0.0028922, 0.0375331, -0.0029075, 0.0377003, -0.0405925, 0.0404406
3: -0.0041084, 0.0682905, -0.0041280, 0.0686103, -0.0655687, 0.0647458
4: -0.0247351, -0.0025060, -0.0247422, -0.0024963, -0.0222388, 0.0222361
5: 0.0054697, 0.0485683, 0.0054508, 0.0487399, -0.0411207, 0.0410181
6: -0.0342384, 0.0513647, -0.0342584, 0.0518458, -0.0860842, 0.0856232
7: -0.0158908, 0.0121729, -0.0159051, 0.0121830, -0.0280738, 0.0280781
8: 0.6969191, 0.9441855, 0.6955412, 0.9442201, -0.2012339, 0.2057638
9: 0.0497447, 0.0928164, 0.0497257, 0.0929809, -0.0432362, 0.0430907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1149167
time: 0.53 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.39 seconds
IS_A1_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B1_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1086967
IS_A1_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1086967
IS_A1_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A2_B1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
IS_A2_B2_A1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1057859
IS_A2_B2_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1057859
IS_A2_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
IS_A2_B2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144414
IS_A2_B2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
IS_A2_B2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1218361
IS_A2_B2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1218361
IS_A2_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.39
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1149167

## BFS IS instance: IS_A1_B1_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0510147, 0.0647588, -0.0483328, 0.0637884, -0.1148030, 0.1130916
1: -0.0190376, 0.0191981, -0.0175171, 0.0174706, -0.0365082, 0.0367152
2: -0.0082249, 0.0389161, -0.0063692, 0.0385845, -0.0468094, 0.0452852
3: -0.0107923, 0.0709348, -0.0084666, 0.0703007, -0.0725117, 0.0711887
4: -0.0271825, 0.0008495, -0.0263309, -0.0003181, -0.0268643, 0.0271804
5: -0.0009829, 0.0499868, 0.0012626, 0.0496467, -0.0506296, 0.0487242
6: -0.0410434, 0.0553416, -0.0386754, 0.0543882, -0.0954316, 0.0940170
7: -0.0207958, 0.0156302, -0.0190889, 0.0144272, -0.0352230, 0.0347191
8: 0.6855294, 0.9560105, 0.6882600, 0.9518959, -0.2172661, 0.2168026
9: 0.0432519, 0.0941757, 0.0455112, 0.0938498, -0.0505979, 0.0486645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0510144, 0.0648657, -0.0478678, 0.0645912, -0.1156056, 0.1127336
1: -0.0190375, 0.0191981, -0.0172537, 0.0171715, -0.0362090, 0.0364518
2: -0.0082250, 0.0389526, -0.0060478, 0.0388587, -0.0470838, 0.0450003
3: -0.0107923, 0.0710048, -0.0080637, 0.0708254, -0.0729987, 0.0711828
4: -0.0271825, 0.0008495, -0.0261833, -0.0005204, -0.0266621, 0.0270329
5: -0.0009829, 0.0500243, 0.0016516, 0.0499281, -0.0506250, 0.0483727
6: -0.0410436, 0.0554467, -0.0382654, 0.0551771, -0.0962207, 0.0937121
7: -0.0207957, 0.0156302, -0.0187933, 0.0142187, -0.0350144, 0.0344235
8: 0.6852283, 0.9560106, 0.6860011, 0.9511826, -0.2189865, 0.2183182
9: 0.0432520, 0.0942116, 0.0459027, 0.0941195, -0.0508675, 0.0483089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0510147, 0.0647588, -0.0436411, 0.0592813, -0.1102960, 0.1083999
1: -0.0190376, 0.0191981, -0.0148572, 0.0144485, -0.0334861, 0.0340553
2: -0.0082249, 0.0389161, -0.0031229, 0.0370446, -0.0452695, 0.0420389
3: -0.0107923, 0.0709348, -0.0043976, 0.0673565, -0.0699745, 0.0672492
4: -0.0271825, 0.0008495, -0.0248410, -0.0023608, -0.0248216, 0.0256906
5: -0.0009829, 0.0499868, 0.0051905, 0.0480673, -0.0476803, 0.0447963
6: -0.0410434, 0.0553416, -0.0345330, 0.0499602, -0.0910036, 0.0898746
7: -0.0207958, 0.0156302, -0.0161031, 0.0123225, -0.0331183, 0.0317333
8: 0.6855294, 0.9560105, 0.7009428, 0.9446972, -0.2109051, 0.2065475
9: 0.0432519, 0.0941757, 0.0494637, 0.0923364, -0.0490845, 0.0447120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0510144, 0.0648657, -0.0432869, 0.0601637, -0.1111781, 0.1081526
1: -0.0190375, 0.0191981, -0.0146567, 0.0142207, -0.0332583, 0.0338549
2: -0.0082250, 0.0389526, -0.0028783, 0.0373461, -0.0455711, 0.0418309
3: -0.0107923, 0.0710048, -0.0040907, 0.0679330, -0.0705031, 0.0672598
4: -0.0271825, 0.0008495, -0.0247287, -0.0025147, -0.0246677, 0.0255782
5: -0.0009829, 0.0500243, 0.0054865, 0.0483766, -0.0473337, 0.0445378
6: -0.0410436, 0.0554467, -0.0342208, 0.0508273, -0.0918709, 0.0896675
7: -0.0207957, 0.0156302, -0.0158781, 0.0121639, -0.0329596, 0.0315083
8: 0.6852283, 0.9560106, 0.6984591, 0.9441545, -0.2123556, 0.2082403
9: 0.0432520, 0.0942116, 0.0497617, 0.0926327, -0.0493808, 0.0444500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0483328, 0.0637884, -0.1101855, 0.1066271
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337410
2: -0.0050301, 0.0367074, -0.0063692, 0.0385845, -0.0436146, 0.0430765
3: -0.0067880, 0.0667118, -0.0084666, 0.0703007, -0.0685304, 0.0669785
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028829, 0.0477215, 0.0012626, 0.0496467, -0.0467639, 0.0464589
6: -0.0369667, 0.0489906, -0.0386754, 0.0543882, -0.0913549, 0.0876660
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326478
8: 0.7037196, 0.9489262, 0.6882600, 0.9518959, -0.1992197, 0.2102108
9: 0.0471417, 0.0920050, 0.0455112, 0.0938498, -0.0467081, 0.0464938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0436411, 0.0592813, -0.1056784, 0.1019353
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050301, 0.0367074, -0.0031229, 0.0370446, -0.0420747, 0.0398302
3: -0.0067880, 0.0667118, -0.0043976, 0.0673565, -0.0663639, 0.0637336
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028829, 0.0477215, 0.0051905, 0.0480673, -0.0440732, 0.0425310
6: -0.0369667, 0.0489906, -0.0345330, 0.0499602, -0.0869269, 0.0835236
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301797, 0.0296620
8: 0.7037196, 0.9489262, 0.7009428, 0.9446972, -0.1967988, 0.2020478
9: 0.0471417, 0.0920050, 0.0494637, 0.0923364, -0.0451947, 0.0425412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0483328, 0.0637884, -0.1101855, 0.1066271
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337410
2: -0.0050301, 0.0367074, -0.0063692, 0.0385845, -0.0436146, 0.0430765
3: -0.0067880, 0.0667118, -0.0084666, 0.0703007, -0.0685304, 0.0669785
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028829, 0.0477215, 0.0012626, 0.0496467, -0.0467639, 0.0464589
6: -0.0369667, 0.0489906, -0.0386754, 0.0543882, -0.0913549, 0.0876660
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326478
8: 0.7037196, 0.9489262, 0.6882600, 0.9518959, -0.1992197, 0.2102108
9: 0.0471417, 0.0920050, 0.0455112, 0.0938498, -0.0467081, 0.0464938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0436411, 0.0592813, -0.1056784, 0.1019353
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050301, 0.0367074, -0.0031229, 0.0370446, -0.0420747, 0.0398302
3: -0.0067880, 0.0667118, -0.0043976, 0.0673565, -0.0663639, 0.0637336
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028829, 0.0477215, 0.0051905, 0.0480673, -0.0440732, 0.0425310
6: -0.0369667, 0.0489906, -0.0345330, 0.0499602, -0.0869269, 0.0835236
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301797, 0.0296620
8: 0.7037196, 0.9489262, 0.7009428, 0.9446972, -0.1967988, 0.2020478
9: 0.0471417, 0.0920050, 0.0494637, 0.0923364, -0.0451947, 0.0425412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0483328, 0.0637884, -0.1101855, 0.1066271
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337410
2: -0.0050301, 0.0367074, -0.0063692, 0.0385845, -0.0436146, 0.0430765
3: -0.0067880, 0.0667118, -0.0084666, 0.0703007, -0.0685304, 0.0669785
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028829, 0.0477215, 0.0012626, 0.0496467, -0.0467639, 0.0464589
6: -0.0369667, 0.0489906, -0.0386754, 0.0543882, -0.0913549, 0.0876660
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326478
8: 0.7037196, 0.9489262, 0.6882600, 0.9518959, -0.1992197, 0.2102108
9: 0.0471417, 0.0920050, 0.0455112, 0.0938498, -0.0467081, 0.0464938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.50 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0483328, 0.0637884, -0.1101855, 0.1066271
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337410
2: -0.0050301, 0.0367074, -0.0063692, 0.0385845, -0.0436146, 0.0430765
3: -0.0067880, 0.0667118, -0.0084666, 0.0703007, -0.0685304, 0.0669785
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028829, 0.0477215, 0.0012626, 0.0496467, -0.0467639, 0.0464589
6: -0.0369667, 0.0489906, -0.0386754, 0.0543882, -0.0913549, 0.0876660
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326478
8: 0.7037196, 0.9489262, 0.6882600, 0.9518959, -0.1992197, 0.2102108
9: 0.0471417, 0.0920050, 0.0455112, 0.0938498, -0.0467081, 0.0464938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0582943, -0.0436411, 0.0592813, -0.1056784, 0.1019353
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050301, 0.0367074, -0.0031229, 0.0370446, -0.0420747, 0.0398302
3: -0.0067880, 0.0667118, -0.0043976, 0.0673565, -0.0663639, 0.0637336
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028829, 0.0477215, 0.0051905, 0.0480673, -0.0440732, 0.0425310
6: -0.0369667, 0.0489906, -0.0345330, 0.0499602, -0.0869269, 0.0835236
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301797, 0.0296620
8: 0.7037196, 0.9489262, 0.7009428, 0.9446972, -0.1967988, 0.2020478
9: 0.0471417, 0.0920050, 0.0494637, 0.0923364, -0.0451947, 0.0425412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0483328, 0.0637884, -0.1102077, 0.1037812
1: -0.0164326, 0.0162384, -0.0175171, 0.0174706, -0.0339032, 0.0337555
2: -0.0050456, 0.0357351, -0.0063692, 0.0385845, -0.0436301, 0.0421042
3: -0.0068073, 0.0648526, -0.0084666, 0.0703007, -0.0683667, 0.0651133
4: -0.0257235, -0.0011510, -0.0263309, -0.0003181, -0.0254053, 0.0251799
5: 0.0028639, 0.0467242, 0.0012626, 0.0496467, -0.0467829, 0.0454616
6: -0.0369866, 0.0461946, -0.0386754, 0.0543882, -0.0913748, 0.0848700
7: -0.0178716, 0.0135691, -0.0190889, 0.0144272, -0.0322988, 0.0326580
8: 0.7117278, 0.9489607, 0.6882600, 0.9518959, -0.1911783, 0.2084191
9: 0.0471227, 0.0910493, 0.0455112, 0.0938498, -0.0467272, 0.0455381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0478678, 0.0645912, -0.1110106, 0.1033163
1: -0.0164326, 0.0162384, -0.0172537, 0.0171715, -0.0336040, 0.0334921
2: -0.0050456, 0.0357351, -0.0060478, 0.0388587, -0.0439044, 0.0417828
3: -0.0068073, 0.0648526, -0.0080637, 0.0708254, -0.0692218, 0.0650368
4: -0.0257235, -0.0011510, -0.0261833, -0.0005204, -0.0252031, 0.0250324
5: 0.0028639, 0.0467242, 0.0016516, 0.0499281, -0.0469098, 0.0450726
6: -0.0369866, 0.0461946, -0.0382654, 0.0551771, -0.0921637, 0.0844600
7: -0.0178716, 0.0135691, -0.0187933, 0.0142187, -0.0320902, 0.0323624
8: 0.7117278, 0.9489607, 0.6860011, 0.9511826, -0.1925888, 0.2129822
9: 0.0471227, 0.0910493, 0.0459027, 0.0941195, -0.0469968, 0.0451466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0436411, 0.0592813, -0.1057007, 0.0990895
1: -0.0164326, 0.0162384, -0.0148572, 0.0144485, -0.0308811, 0.0310956
2: -0.0050456, 0.0357351, -0.0031229, 0.0370446, -0.0420903, 0.0388579
3: -0.0068073, 0.0648526, -0.0043976, 0.0673565, -0.0662386, 0.0618742
4: -0.0257235, -0.0011510, -0.0248410, -0.0023608, -0.0233626, 0.0236900
5: 0.0028639, 0.0467242, 0.0051905, 0.0480673, -0.0439929, 0.0415337
6: -0.0369866, 0.0461946, -0.0345330, 0.0499602, -0.0869468, 0.0807276
7: -0.0178716, 0.0135691, -0.0161031, 0.0123225, -0.0301941, 0.0296722
8: 0.7117278, 0.9489607, 0.7009428, 0.9446972, -0.1887765, 0.2004530
9: 0.0471227, 0.0910493, 0.0494637, 0.0923364, -0.0452137, 0.0415856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1086641
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0483328, 0.0637884, -0.1101850, 0.1028940
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337410
2: -0.0050301, 0.0354300, -0.0063692, 0.0385845, -0.0436146, 0.0417992
3: -0.0067881, 0.0642716, -0.0084666, 0.0703007, -0.0685303, 0.0647877
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0464113, 0.0012626, 0.0496467, -0.0467640, 0.0451487
6: -0.0369667, 0.0453175, -0.0386754, 0.0543882, -0.0913549, 0.0839930
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7142399, 0.9489262, 0.6882600, 0.9518959, -0.1904554, 0.2102029
9: 0.0471418, 0.0907734, 0.0455112, 0.0938498, -0.0467080, 0.0452622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0478678, 0.0645912, -0.1109878, 0.1024290
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334776
2: -0.0050301, 0.0354300, -0.0060478, 0.0388587, -0.0438888, 0.0414778
3: -0.0067881, 0.0642716, -0.0080637, 0.0708254, -0.0690598, 0.0643206
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0464113, 0.0016516, 0.0499281, -0.0468019, 0.0447597
6: -0.0369667, 0.0453175, -0.0382654, 0.0551771, -0.0921438, 0.0835829
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320758, 0.0323523
8: 0.7142399, 0.9489262, 0.6860011, 0.9511826, -0.1888852, 0.2116635
9: 0.0471418, 0.0907734, 0.0459027, 0.0941195, -0.0469777, 0.0448707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0436411, 0.0592813, -0.1056779, 0.0982022
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050301, 0.0354300, -0.0031229, 0.0370446, -0.0420747, 0.0385529
3: -0.0067881, 0.0642716, -0.0043976, 0.0673565, -0.0663624, 0.0615070
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0464113, 0.0051905, 0.0480673, -0.0440713, 0.0412208
6: -0.0369667, 0.0453175, -0.0345330, 0.0499602, -0.0869269, 0.0798505
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301797, 0.0296620
8: 0.7142399, 0.9489262, 0.7009428, 0.9446972, -0.1879029, 0.2020402
9: 0.0471418, 0.0907734, 0.0494637, 0.0923364, -0.0451946, 0.0413097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1086641
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0483788, 0.0643713, -0.1076785, 0.1121537
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0385798, -0.0064011, 0.0387837, -0.0416758, 0.0449809
3: -0.0041084, 0.0702920, -0.0085063, 0.0706818, -0.0669177, 0.0705454
4: -0.0247350, -0.0024921, -0.0263454, -0.0002981, -0.0244369, 0.0238533
5: 0.0054697, 0.0496420, 0.0012239, 0.0498511, -0.0443814, 0.0461946
6: -0.0342384, 0.0543750, -0.0387161, 0.0549611, -0.0891995, 0.0930911
7: -0.0158908, 0.0121728, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6882984, 0.9441848, 0.6866189, 0.9519659, -0.2143590, 0.2108366
9: 0.0497448, 0.0938453, 0.0454726, 0.0940456, -0.0443008, 0.0483727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0436862, 0.0600004, -0.1033075, 0.1074611
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0385798, -0.0031544, 0.0372902, -0.0401823, 0.0417342
3: -0.0041084, 0.0702920, -0.0044372, 0.0678261, -0.0648525, 0.0671834
4: -0.0247350, -0.0024921, -0.0248554, -0.0023410, -0.0223940, 0.0223633
5: 0.0054697, 0.0496420, 0.0051522, 0.0483193, -0.0414594, 0.0418269
6: -0.0342384, 0.0543750, -0.0345731, 0.0506667, -0.0849051, 0.0889481
7: -0.0158908, 0.0121728, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6882984, 0.9441848, 0.6989187, 0.9447667, -0.2113264, 0.2029898
9: 0.0497448, 0.0938453, 0.0494252, 0.0925778, -0.0428330, 0.0444201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.52 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0483788, 0.0643713, -0.1076785, 0.1121537
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0385798, -0.0064011, 0.0387837, -0.0416758, 0.0449809
3: -0.0041084, 0.0702920, -0.0085063, 0.0706818, -0.0669177, 0.0705454
4: -0.0247350, -0.0024921, -0.0263454, -0.0002981, -0.0244369, 0.0238533
5: 0.0054697, 0.0496420, 0.0012239, 0.0498511, -0.0443814, 0.0461946
6: -0.0342384, 0.0543750, -0.0387161, 0.0549611, -0.0891995, 0.0930911
7: -0.0158908, 0.0121728, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6882984, 0.9441848, 0.6866189, 0.9519659, -0.2143590, 0.2108366
9: 0.0497448, 0.0938453, 0.0454726, 0.0940456, -0.0443008, 0.0483727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0436862, 0.0600004, -0.1033075, 0.1074611
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0385798, -0.0031544, 0.0372902, -0.0401823, 0.0417342
3: -0.0041084, 0.0702920, -0.0044372, 0.0678261, -0.0648525, 0.0671834
4: -0.0247350, -0.0024921, -0.0248554, -0.0023410, -0.0223940, 0.0223633
5: 0.0054697, 0.0496420, 0.0051522, 0.0483193, -0.0414594, 0.0418269
6: -0.0342384, 0.0543750, -0.0345731, 0.0506667, -0.0849051, 0.0889481
7: -0.0158908, 0.0121728, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6882984, 0.9441848, 0.6989187, 0.9447667, -0.2113264, 0.2029898
9: 0.0497448, 0.0938453, 0.0494252, 0.0925778, -0.0428330, 0.0444201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0483788, 0.0643713, -0.1080337, 0.1079780
1: -0.0148696, 0.0144624, -0.0175432, 0.0175002, -0.0323698, 0.0320056
2: -0.0031379, 0.0371533, -0.0064011, 0.0387837, -0.0419216, 0.0435544
3: -0.0044166, 0.0675643, -0.0085063, 0.0706818, -0.0669786, 0.0677458
4: -0.0248479, -0.0023514, -0.0263454, -0.0002981, -0.0245498, 0.0239940
5: 0.0051722, 0.0481788, 0.0012239, 0.0498511, -0.0446789, 0.0458507
6: -0.0345522, 0.0502727, -0.0387161, 0.0549611, -0.0895134, 0.0889888
7: -0.0161169, 0.0123323, -0.0191182, 0.0144478, -0.0305647, 0.0314505
8: 0.7000468, 0.9447302, 0.6866189, 0.9519659, -0.2021253, 0.2096937
9: 0.0494456, 0.0924433, 0.0454726, 0.0940456, -0.0446001, 0.0469706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047682, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0436862, 0.0600004, -0.1036627, 0.1032854
1: -0.0148696, 0.0144624, -0.0148830, 0.0144780, -0.0293476, 0.0293454
2: -0.0031379, 0.0371533, -0.0031544, 0.0372902, -0.0404281, 0.0403077
3: -0.0044166, 0.0675643, -0.0044372, 0.0678261, -0.0648879, 0.0643740
4: -0.0248479, -0.0023514, -0.0248554, -0.0023410, -0.0225069, 0.0225040
5: 0.0051722, 0.0481788, 0.0051522, 0.0483193, -0.0415780, 0.0415182
6: -0.0345522, 0.0502727, -0.0345731, 0.0506667, -0.0852190, 0.0848458
7: -0.0161169, 0.0123323, -0.0161321, 0.0123429, -0.0284598, 0.0284644
8: 0.7000468, 0.9447302, 0.6989187, 0.9447667, -0.1989958, 0.2017820
9: 0.0494456, 0.0924433, 0.0494252, 0.0925778, -0.0431323, 0.0430180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047682, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0479112, 0.0654786, -0.1091410, 0.1075104
1: -0.0148696, 0.0144624, -0.0172781, 0.0171990, -0.0320686, 0.0317405
2: -0.0031379, 0.0371533, -0.0060775, 0.0391620, -0.0422999, 0.0432308
3: -0.0044166, 0.0675643, -0.0081012, 0.0714051, -0.0680443, 0.0676265
4: -0.0248479, -0.0023514, -0.0261970, -0.0005017, -0.0243462, 0.0238456
5: 0.0051722, 0.0481788, 0.0016154, 0.0502390, -0.0445277, 0.0456468
6: -0.0345522, 0.0502727, -0.0383031, 0.0560488, -0.0906010, 0.0885758
7: -0.0161169, 0.0123323, -0.0188207, 0.0142381, -0.0303550, 0.0311530
8: 0.7000468, 0.9447302, 0.6835042, 0.9512478, -0.2033772, 0.2150698
9: 0.0494456, 0.0924433, 0.0458664, 0.0944174, -0.0449719, 0.0465768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0433297, 0.0612006, -0.1048630, 0.1029289
1: -0.0148696, 0.0144624, -0.0146810, 0.0142479, -0.0291175, 0.0291434
2: -0.0031379, 0.0371533, -0.0029075, 0.0377003, -0.0408382, 0.0400608
3: -0.0044166, 0.0675643, -0.0041280, 0.0686103, -0.0659669, 0.0642941
4: -0.0248479, -0.0023514, -0.0247422, -0.0024963, -0.0223516, 0.0223908
5: 0.0051722, 0.0481788, 0.0054508, 0.0487399, -0.0416168, 0.0413705
6: -0.0345522, 0.0502727, -0.0342584, 0.0518458, -0.0863980, 0.0845311
7: -0.0161169, 0.0123323, -0.0159051, 0.0121830, -0.0282999, 0.0282374
8: 0.7000468, 0.9447302, 0.6955412, 0.9442201, -0.2000279, 0.2073538
9: 0.0494456, 0.0924433, 0.0497257, 0.0929809, -0.0435354, 0.0427175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0483788, 0.0643713, -0.1076781, 0.1090898
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321682, 0.0317768
2: -0.0028922, 0.0375331, -0.0064011, 0.0387837, -0.0416758, 0.0439342
3: -0.0041084, 0.0682905, -0.0085063, 0.0706818, -0.0669172, 0.0687420
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054697, 0.0485683, 0.0012239, 0.0498511, -0.0443814, 0.0458016
6: -0.0342384, 0.0513647, -0.0387161, 0.0549611, -0.0891995, 0.0900808
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6969191, 0.9441855, 0.6866189, 0.9519659, -0.2072566, 0.2108314
9: 0.0497447, 0.0928164, 0.0454726, 0.0940456, -0.0443009, 0.0473438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0436862, 0.0600004, -0.1033072, 0.1043972
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028922, 0.0375331, -0.0031544, 0.0372902, -0.0401824, 0.0406874
3: -0.0041084, 0.0682905, -0.0044372, 0.0678261, -0.0648521, 0.0653646
4: -0.0247351, -0.0025060, -0.0248554, -0.0023410, -0.0223941, 0.0223494
5: 0.0054697, 0.0485683, 0.0051522, 0.0483193, -0.0414588, 0.0414622
6: -0.0342384, 0.0513647, -0.0345731, 0.0506667, -0.0849051, 0.0859378
7: -0.0158908, 0.0121729, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6969191, 0.9441855, 0.6989187, 0.9447667, -0.2041392, 0.2029879
9: 0.0497447, 0.0928164, 0.0494252, 0.0925778, -0.0428331, 0.0433912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0479112, 0.0654786, -0.1087854, 0.1086222
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315116
2: -0.0028922, 0.0375331, -0.0060775, 0.0391620, -0.0420541, 0.0436105
3: -0.0041084, 0.0682905, -0.0081012, 0.0714051, -0.0675853, 0.0680267
4: -0.0247351, -0.0025060, -0.0261970, -0.0005017, -0.0242334, 0.0236909
5: 0.0054697, 0.0485683, 0.0016154, 0.0502390, -0.0441304, 0.0452747
6: -0.0342384, 0.0513647, -0.0383031, 0.0560488, -0.0902872, 0.0896678
7: -0.0158908, 0.0121729, -0.0188207, 0.0142381, -0.0301289, 0.0309936
8: 0.6969191, 0.9441855, 0.6835042, 0.9512478, -0.2042236, 0.2132380
9: 0.0497447, 0.0928164, 0.0458664, 0.0944174, -0.0446727, 0.0469500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0433297, 0.0612006, -0.1045074, 0.1040407
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289145
2: -0.0028922, 0.0375331, -0.0029075, 0.0377003, -0.0405925, 0.0404406
3: -0.0041084, 0.0682905, -0.0041280, 0.0686103, -0.0655687, 0.0647458
4: -0.0247351, -0.0025060, -0.0247422, -0.0024963, -0.0222388, 0.0222361
5: 0.0054697, 0.0485683, 0.0054508, 0.0487399, -0.0411207, 0.0410181
6: -0.0342384, 0.0513647, -0.0342584, 0.0518458, -0.0860842, 0.0856232
7: -0.0158908, 0.0121729, -0.0159051, 0.0121830, -0.0280738, 0.0280781
8: 0.6969191, 0.9441855, 0.6955412, 0.9442201, -0.2012339, 0.2057638
9: 0.0497447, 0.0928164, 0.0497257, 0.0929809, -0.0432362, 0.0430907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0483788, 0.0643713, -0.1122595, 0.1164205
1: -0.0172653, 0.0171843, -0.0175432, 0.0175002, -0.0347654, 0.0347275
2: -0.0060618, 0.0400376, -0.0064011, 0.0387837, -0.0448455, 0.0464387
3: -0.0080811, 0.0730793, -0.0085063, 0.0706818, -0.0708404, 0.0733056
4: -0.0261897, -0.0005116, -0.0263454, -0.0002981, -0.0258916, 0.0258338
5: 0.0016343, 0.0511372, 0.0012239, 0.0498511, -0.0482168, 0.0486927
6: -0.0382830, 0.0585671, -0.0387161, 0.0549611, -0.0932441, 0.0972831
7: -0.0188062, 0.0142279, -0.0191182, 0.0144478, -0.0332540, 0.0333461
8: 0.6762923, 0.9512142, 0.6866189, 0.9519659, -0.2261345, 0.2174623
9: 0.0458855, 0.0952781, 0.0454726, 0.0940456, -0.0481602, 0.0498055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0436862, 0.0600004, -0.1078885, 0.1117279
1: -0.0172653, 0.0171843, -0.0148830, 0.0144780, -0.0317432, 0.0320673
2: -0.0060618, 0.0400376, -0.0031544, 0.0372902, -0.0433521, 0.0431920
3: -0.0080811, 0.0730793, -0.0044372, 0.0678261, -0.0680933, 0.0693383
4: -0.0261897, -0.0005116, -0.0248554, -0.0023410, -0.0238487, 0.0243439
5: 0.0016343, 0.0511372, 0.0051522, 0.0483193, -0.0457268, 0.0448328
6: -0.0382830, 0.0585671, -0.0345731, 0.0506667, -0.0889497, 0.0931402
7: -0.0188062, 0.0142279, -0.0161321, 0.0123429, -0.0311491, 0.0303600
8: 0.6762923, 0.9512142, 0.6989187, 0.9447667, -0.2196519, 0.2058556
9: 0.0458855, 0.0952781, 0.0494252, 0.0925778, -0.0466924, 0.0458529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0479110, 0.0671977, -0.1150859, 0.1159527
1: -0.0172653, 0.0171843, -0.0172780, 0.0171990, -0.0344643, 0.0344623
2: -0.0060618, 0.0400376, -0.0060776, 0.0397492, -0.0458111, 0.0461152
3: -0.0080811, 0.0730793, -0.0081008, 0.0725281, -0.0726549, 0.0731865
4: -0.0261897, -0.0005116, -0.0261969, -0.0005017, -0.0256880, 0.0256853
5: 0.0016343, 0.0511372, 0.0016153, 0.0508415, -0.0485761, 0.0484890
6: -0.0382830, 0.0585671, -0.0383031, 0.0577376, -0.0960205, 0.0968701
7: -0.0188062, 0.0142279, -0.0188207, 0.0142381, -0.0330442, 0.0330486
8: 0.6762923, 0.9512142, 0.6786667, 0.9512488, -0.2273870, 0.2248356
9: 0.0458855, 0.0952781, 0.0458664, 0.0949947, -0.0491092, 0.0494117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479108, 0.0662074, -0.1095146, 0.1116857
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0394109, -0.0423031, 0.0446573
3: -0.0041084, 0.0702920, -0.0081008, 0.0718812, -0.0680928, 0.0706359
4: -0.0247350, -0.0024921, -0.0261969, -0.0005017, -0.0242333, 0.0237048
5: 0.0054697, 0.0496420, 0.0016155, 0.0504945, -0.0447388, 0.0452268
6: -0.0342384, 0.0543750, -0.0383035, 0.0567648, -0.0910032, 0.0926785
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6814528, 0.9512495, -0.2168379, 0.2154725
9: 0.0497448, 0.0938453, 0.0458663, 0.0946622, -0.0449174, 0.0479790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1047672
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0483788, 0.0643713, -0.1076785, 0.1121537
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0385798, -0.0064011, 0.0387837, -0.0416758, 0.0449809
3: -0.0041084, 0.0702920, -0.0085063, 0.0706818, -0.0669177, 0.0705454
4: -0.0247350, -0.0024921, -0.0263454, -0.0002981, -0.0244369, 0.0238533
5: 0.0054697, 0.0496420, 0.0012239, 0.0498511, -0.0443814, 0.0461946
6: -0.0342384, 0.0543750, -0.0387161, 0.0549611, -0.0891995, 0.0930911
7: -0.0158908, 0.0121728, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6882984, 0.9441848, 0.6866189, 0.9519659, -0.2143590, 0.2108366
9: 0.0497448, 0.0938453, 0.0454726, 0.0940456, -0.0443008, 0.0483727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0436862, 0.0600004, -0.1033075, 0.1074611
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0385798, -0.0031544, 0.0372902, -0.0401823, 0.0417342
3: -0.0041084, 0.0702920, -0.0044372, 0.0678261, -0.0648525, 0.0671834
4: -0.0247350, -0.0024921, -0.0248554, -0.0023410, -0.0223940, 0.0223633
5: 0.0054697, 0.0496420, 0.0051522, 0.0483193, -0.0414594, 0.0418269
6: -0.0342384, 0.0543750, -0.0345731, 0.0506667, -0.0849051, 0.0889481
7: -0.0158908, 0.0121728, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6882984, 0.9441848, 0.6989187, 0.9447667, -0.2113264, 0.2029898
9: 0.0497448, 0.0938453, 0.0494252, 0.0925778, -0.0428330, 0.0444201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0479112, 0.0654786, -0.1133668, 0.1159529
1: -0.0172653, 0.0171843, -0.0172781, 0.0171990, -0.0344642, 0.0344624
2: -0.0060618, 0.0400376, -0.0060775, 0.0391620, -0.0452238, 0.0461151
3: -0.0080811, 0.0730793, -0.0081012, 0.0714051, -0.0715061, 0.0731862
4: -0.0261897, -0.0005116, -0.0261970, -0.0005017, -0.0256880, 0.0256854
5: 0.0016343, 0.0511372, 0.0016154, 0.0502390, -0.0479315, 0.0484888
6: -0.0382830, 0.0585671, -0.0383031, 0.0560488, -0.0943318, 0.0968702
7: -0.0188062, 0.0142279, -0.0188207, 0.0142381, -0.0330443, 0.0330486
8: 0.6762923, 0.9512142, 0.6835042, 0.9512478, -0.2273865, 0.2198558
9: 0.0458855, 0.0952781, 0.0458664, 0.0944174, -0.0485320, 0.0494117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0483788, 0.0643713, -0.1076785, 0.1121537
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0385798, -0.0064011, 0.0387837, -0.0416758, 0.0449809
3: -0.0041084, 0.0702920, -0.0085063, 0.0706818, -0.0669177, 0.0705454
4: -0.0247350, -0.0024921, -0.0263454, -0.0002981, -0.0244369, 0.0238533
5: 0.0054697, 0.0496420, 0.0012239, 0.0498511, -0.0443814, 0.0461946
6: -0.0342384, 0.0543750, -0.0387161, 0.0549611, -0.0891995, 0.0930911
7: -0.0158908, 0.0121728, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6882984, 0.9441848, 0.6866189, 0.9519659, -0.2143590, 0.2108366
9: 0.0497448, 0.0938453, 0.0454726, 0.0940456, -0.0443008, 0.0483727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0436862, 0.0600004, -0.1033075, 0.1074611
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0385798, -0.0031544, 0.0372902, -0.0401823, 0.0417342
3: -0.0041084, 0.0702920, -0.0044372, 0.0678261, -0.0648525, 0.0671834
4: -0.0247350, -0.0024921, -0.0248554, -0.0023410, -0.0223940, 0.0223633
5: 0.0054697, 0.0496420, 0.0051522, 0.0483193, -0.0414594, 0.0418269
6: -0.0342384, 0.0543750, -0.0345731, 0.0506667, -0.0849051, 0.0889481
7: -0.0158908, 0.0121728, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6882984, 0.9441848, 0.6989187, 0.9447667, -0.2113264, 0.2029898
9: 0.0497448, 0.0938453, 0.0494252, 0.0925778, -0.0428330, 0.0444201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0433297, 0.0612006, -0.1045078, 0.1071046
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289146
2: -0.0028921, 0.0385798, -0.0029075, 0.0377003, -0.0405924, 0.0414873
3: -0.0041084, 0.0702920, -0.0041280, 0.0686103, -0.0655686, 0.0671036
4: -0.0247350, -0.0024921, -0.0247422, -0.0024963, -0.0222387, 0.0222501
5: 0.0054697, 0.0496420, 0.0054508, 0.0487399, -0.0411208, 0.0416791
6: -0.0342384, 0.0543750, -0.0342584, 0.0518458, -0.0860841, 0.0886334
7: -0.0158908, 0.0121728, -0.0159051, 0.0121830, -0.0280738, 0.0280780
8: 0.6882984, 0.9441848, 0.6955412, 0.9442201, -0.2123580, 0.2057641
9: 0.0497448, 0.0938453, 0.0497257, 0.0929809, -0.0432361, 0.0441196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0483788, 0.0643713, -0.1076785, 0.1121537
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0385798, -0.0064011, 0.0387837, -0.0416758, 0.0449809
3: -0.0041084, 0.0702920, -0.0085063, 0.0706818, -0.0669177, 0.0705454
4: -0.0247350, -0.0024921, -0.0263454, -0.0002981, -0.0244369, 0.0238533
5: 0.0054697, 0.0496420, 0.0012239, 0.0498511, -0.0443814, 0.0461946
6: -0.0342384, 0.0543750, -0.0387161, 0.0549611, -0.0891995, 0.0930911
7: -0.0158908, 0.0121728, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6882984, 0.9441848, 0.6866189, 0.9519659, -0.2143590, 0.2108366
9: 0.0497448, 0.0938453, 0.0454726, 0.0940456, -0.0443008, 0.0483727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0436862, 0.0600004, -0.1033075, 0.1074611
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0385798, -0.0031544, 0.0372902, -0.0401823, 0.0417342
3: -0.0041084, 0.0702920, -0.0044372, 0.0678261, -0.0648525, 0.0671834
4: -0.0247350, -0.0024921, -0.0248554, -0.0023410, -0.0223940, 0.0223633
5: 0.0054697, 0.0496420, 0.0051522, 0.0483193, -0.0414594, 0.0418269
6: -0.0342384, 0.0543750, -0.0345731, 0.0506667, -0.0849051, 0.0889481
7: -0.0158908, 0.0121728, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6882984, 0.9441848, 0.6989187, 0.9447667, -0.2113264, 0.2029898
9: 0.0497448, 0.0938453, 0.0494252, 0.0925778, -0.0428330, 0.0444201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0478882, 0.0680417, -0.0479112, 0.0654786, -0.1133668, 0.1159529
1: -0.0172653, 0.0171843, -0.0172781, 0.0171990, -0.0344642, 0.0344624
2: -0.0060618, 0.0400376, -0.0060775, 0.0391620, -0.0452238, 0.0461151
3: -0.0080811, 0.0730793, -0.0081012, 0.0714051, -0.0715061, 0.0731862
4: -0.0261897, -0.0005116, -0.0261970, -0.0005017, -0.0256880, 0.0256854
5: 0.0016343, 0.0511372, 0.0016154, 0.0502390, -0.0479315, 0.0484888
6: -0.0382830, 0.0585671, -0.0383031, 0.0560488, -0.0943318, 0.0968702
7: -0.0188062, 0.0142279, -0.0188207, 0.0142381, -0.0330443, 0.0330486
8: 0.6762923, 0.9512142, 0.6835042, 0.9512478, -0.2273865, 0.2198558
9: 0.0458855, 0.0952781, 0.0458664, 0.0944174, -0.0485320, 0.0494117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1057859
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0479112, 0.0654786, -0.1091410, 0.1075104
1: -0.0148696, 0.0144624, -0.0172781, 0.0171990, -0.0320686, 0.0317405
2: -0.0031379, 0.0371533, -0.0060775, 0.0391620, -0.0422999, 0.0432308
3: -0.0044166, 0.0675643, -0.0081012, 0.0714051, -0.0680443, 0.0676265
4: -0.0248479, -0.0023514, -0.0261970, -0.0005017, -0.0243462, 0.0238456
5: 0.0051722, 0.0481788, 0.0016154, 0.0502390, -0.0445277, 0.0456468
6: -0.0345522, 0.0502727, -0.0383031, 0.0560488, -0.0906010, 0.0885758
7: -0.0161169, 0.0123323, -0.0188207, 0.0142381, -0.0303550, 0.0311530
8: 0.7000468, 0.9447302, 0.6835042, 0.9512478, -0.2033772, 0.2150698
9: 0.0494456, 0.0924433, 0.0458664, 0.0944174, -0.0449719, 0.0465768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0433297, 0.0612006, -0.1048630, 0.1029289
1: -0.0148696, 0.0144624, -0.0146810, 0.0142479, -0.0291175, 0.0291434
2: -0.0031379, 0.0371533, -0.0029075, 0.0377003, -0.0408382, 0.0400608
3: -0.0044166, 0.0675643, -0.0041280, 0.0686103, -0.0659669, 0.0642941
4: -0.0248479, -0.0023514, -0.0247422, -0.0024963, -0.0223516, 0.0223908
5: 0.0051722, 0.0481788, 0.0054508, 0.0487399, -0.0416168, 0.0413705
6: -0.0345522, 0.0502727, -0.0342584, 0.0518458, -0.0863980, 0.0845311
7: -0.0161169, 0.0123323, -0.0159051, 0.0121830, -0.0282999, 0.0282374
8: 0.7000468, 0.9447302, 0.6955412, 0.9442201, -0.2000279, 0.2073538
9: 0.0494456, 0.0924433, 0.0497257, 0.0929809, -0.0435354, 0.0427175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0479112, 0.0654786, -0.1087854, 0.1086222
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315116
2: -0.0028922, 0.0375331, -0.0060775, 0.0391620, -0.0420541, 0.0436105
3: -0.0041084, 0.0682905, -0.0081012, 0.0714051, -0.0675853, 0.0680267
4: -0.0247351, -0.0025060, -0.0261970, -0.0005017, -0.0242334, 0.0236909
5: 0.0054697, 0.0485683, 0.0016154, 0.0502390, -0.0441304, 0.0452747
6: -0.0342384, 0.0513647, -0.0383031, 0.0560488, -0.0902872, 0.0896678
7: -0.0158908, 0.0121729, -0.0188207, 0.0142381, -0.0301289, 0.0309936
8: 0.6969191, 0.9441855, 0.6835042, 0.9512478, -0.2042236, 0.2132380
9: 0.0497447, 0.0928164, 0.0458664, 0.0944174, -0.0446727, 0.0469500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144026
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0433297, 0.0612006, -0.1045074, 0.1040407
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289145
2: -0.0028922, 0.0375331, -0.0029075, 0.0377003, -0.0405925, 0.0404406
3: -0.0041084, 0.0682905, -0.0041280, 0.0686103, -0.0655687, 0.0647458
4: -0.0247351, -0.0025060, -0.0247422, -0.0024963, -0.0222388, 0.0222361
5: 0.0054697, 0.0485683, 0.0054508, 0.0487399, -0.0411207, 0.0410181
6: -0.0342384, 0.0513647, -0.0342584, 0.0518458, -0.0860842, 0.0856232
7: -0.0158908, 0.0121729, -0.0159051, 0.0121830, -0.0280738, 0.0280781
8: 0.6969191, 0.9441855, 0.6955412, 0.9442201, -0.2012339, 0.2057638
9: 0.0497447, 0.0928164, 0.0497257, 0.0929809, -0.0432362, 0.0430907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144026
time: 0.54 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.49 seconds
IS_A1_B1_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B1_B1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B1_B1_B1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B1_B2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
IS_A1_B1_B2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B2_B1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B2_B1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B2_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B2_B2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
IS_A1_B2_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B2_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1211631
IS_A1_B2_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1086641
IS_A2_B1_B1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047682, upper bound: 0.1144414
IS_A2_B1_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047682, upper bound: 0.1144414
IS_A2_B1_B2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A1_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B1_B2_B2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
IS_A2_B2_A2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
IS_A2_B2_A2_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
IS_A2_B2_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1057859
IS_A2_B2_A2_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1047672
IS_A2_B2_A2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B2_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B2_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144026
IS_A2_B2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144026

## BFS IS instance: IS_A1_B1_B1_B1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0510148, 0.0648533, -0.0483328, 0.0637884, -0.1148032, 0.1131861
1: -0.0190376, 0.0191981, -0.0175171, 0.0174706, -0.0365082, 0.0367152
2: -0.0082249, 0.0389483, -0.0063692, 0.0385845, -0.0468094, 0.0453174
3: -0.0107924, 0.0709966, -0.0084666, 0.0703007, -0.0725119, 0.0712514
4: -0.0271825, 0.0008495, -0.0263309, -0.0003181, -0.0268644, 0.0271804
5: -0.0009829, 0.0500200, 0.0012626, 0.0496467, -0.0506297, 0.0487574
6: -0.0410436, 0.0554346, -0.0386754, 0.0543882, -0.0954317, 0.0941101
7: -0.0207958, 0.0156302, -0.0190889, 0.0144272, -0.0352230, 0.0347191
8: 0.6852626, 0.9560105, 0.6882600, 0.9518959, -0.2175426, 0.2168024
9: 0.0432518, 0.0942075, 0.0455112, 0.0938498, -0.0505980, 0.0486963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
time: 0.51 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0510145, 0.0641234, -0.0436411, 0.0592813, -0.1102958, 0.1077645
1: -0.0190375, 0.0191981, -0.0148572, 0.0144485, -0.0334861, 0.0340553
2: -0.0082251, 0.0386989, -0.0031229, 0.0370446, -0.0452697, 0.0418218
3: -0.0107923, 0.0705199, -0.0043976, 0.0673565, -0.0699743, 0.0668331
4: -0.0271825, 0.0008495, -0.0248410, -0.0023608, -0.0248217, 0.0256905
5: -0.0009828, 0.0497642, 0.0051905, 0.0480673, -0.0476800, 0.0445738
6: -0.0410434, 0.0547176, -0.0345330, 0.0499602, -0.0910036, 0.0892506
7: -0.0207957, 0.0156302, -0.0161031, 0.0123225, -0.0331183, 0.0317333
8: 0.6873170, 0.9560101, 0.7009428, 0.9446972, -0.2091103, 0.2065470
9: 0.0432518, 0.0939624, 0.0494637, 0.0923364, -0.0490846, 0.0444987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.66 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0510148, 0.0634804, -0.0478678, 0.0645912, -0.1156060, 0.1113483
1: -0.0190375, 0.0191981, -0.0172537, 0.0171715, -0.0362090, 0.0364518
2: -0.0082250, 0.0384792, -0.0060478, 0.0388587, -0.0470838, 0.0445270
3: -0.0107923, 0.0700997, -0.0080637, 0.0708254, -0.0729988, 0.0702797
4: -0.0271825, 0.0008495, -0.0261833, -0.0005204, -0.0266621, 0.0270329
5: -0.0009829, 0.0495388, 0.0016516, 0.0499281, -0.0506250, 0.0478873
6: -0.0410433, 0.0540856, -0.0382654, 0.0551771, -0.0962204, 0.0923510
7: -0.0207958, 0.0156302, -0.0187933, 0.0142187, -0.0350145, 0.0344236
8: 0.6891262, 0.9560096, 0.6860011, 0.9511826, -0.2151041, 0.2183177
9: 0.0432520, 0.0937465, 0.0459027, 0.0941195, -0.0508675, 0.0478438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.69 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0478678, 0.0645912, -0.1110106, 0.1033163
1: -0.0164326, 0.0162384, -0.0172537, 0.0171715, -0.0336040, 0.0334921
2: -0.0050456, 0.0357351, -0.0060478, 0.0388587, -0.0439044, 0.0417828
3: -0.0068073, 0.0648526, -0.0080637, 0.0708254, -0.0692218, 0.0650368
4: -0.0257235, -0.0011510, -0.0261833, -0.0005204, -0.0252031, 0.0250324
5: 0.0028639, 0.0467242, 0.0016516, 0.0499281, -0.0469098, 0.0450726
6: -0.0369866, 0.0461946, -0.0382654, 0.0551771, -0.0921637, 0.0844600
7: -0.0178716, 0.0135691, -0.0187933, 0.0142187, -0.0320902, 0.0323624
8: 0.7117278, 0.9489607, 0.6860011, 0.9511826, -0.1925888, 0.2129822
9: 0.0471227, 0.0910493, 0.0459027, 0.0941195, -0.0469968, 0.0451466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.60 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0478678, 0.0645912, -0.1109878, 0.1024290
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334776
2: -0.0050301, 0.0354300, -0.0060478, 0.0388587, -0.0438888, 0.0414778
3: -0.0067881, 0.0642716, -0.0080637, 0.0708254, -0.0690598, 0.0643206
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0464113, 0.0016516, 0.0499281, -0.0468019, 0.0447597
6: -0.0369667, 0.0453175, -0.0382654, 0.0551771, -0.0921438, 0.0835829
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320758, 0.0323523
8: 0.7142399, 0.9489262, 0.6860011, 0.9511826, -0.1888852, 0.2116635
9: 0.0471418, 0.0907734, 0.0459027, 0.0941195, -0.0469777, 0.0448707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.56 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.59 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0478678, 0.0645912, -0.1110106, 0.1033163
1: -0.0164326, 0.0162384, -0.0172537, 0.0171715, -0.0336040, 0.0334921
2: -0.0050456, 0.0357351, -0.0060478, 0.0388587, -0.0439044, 0.0417828
3: -0.0068073, 0.0648526, -0.0080637, 0.0708254, -0.0692218, 0.0650368
4: -0.0257235, -0.0011510, -0.0261833, -0.0005204, -0.0252031, 0.0250324
5: 0.0028639, 0.0467242, 0.0016516, 0.0499281, -0.0469098, 0.0450726
6: -0.0369866, 0.0461946, -0.0382654, 0.0551771, -0.0921637, 0.0844600
7: -0.0178716, 0.0135691, -0.0187933, 0.0142187, -0.0320902, 0.0323624
8: 0.7117278, 0.9489607, 0.6860011, 0.9511826, -0.1925888, 0.2129822
9: 0.0471227, 0.0910493, 0.0459027, 0.0941195, -0.0469968, 0.0451466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.53 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1086967
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.58 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1086967
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0478678, 0.0645912, -0.1109878, 0.1024290
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334776
2: -0.0050301, 0.0354300, -0.0060478, 0.0388587, -0.0438888, 0.0414778
3: -0.0067881, 0.0642716, -0.0080637, 0.0708254, -0.0690598, 0.0643206
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0464113, 0.0016516, 0.0499281, -0.0468019, 0.0447597
6: -0.0369667, 0.0453175, -0.0382654, 0.0551771, -0.0921438, 0.0835829
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320758, 0.0323523
8: 0.7142399, 0.9489262, 0.6860011, 0.9511826, -0.1888852, 0.2116635
9: 0.0471418, 0.0907734, 0.0459027, 0.0941195, -0.0469777, 0.0448707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.54 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0478678, 0.0645912, -0.1109883, 0.1062692
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334775
2: -0.0050299, 0.0367439, -0.0060478, 0.0388587, -0.0438887, 0.0427917
3: -0.0067881, 0.0667817, -0.0080637, 0.0708254, -0.0690598, 0.0669726
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0477590, 0.0016516, 0.0499281, -0.0468021, 0.0461075
6: -0.0369665, 0.0490958, -0.0382654, 0.0551771, -0.0921436, 0.0873612
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320759, 0.0323523
8: 0.7034178, 0.9489263, 0.6860011, 0.9511826, -0.2009430, 0.2116647
9: 0.0471417, 0.0920409, 0.0459027, 0.0941195, -0.0469778, 0.0461382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0483328, 0.0637884, -0.1101855, 0.1067341
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337409
2: -0.0050299, 0.0367439, -0.0063692, 0.0385845, -0.0436145, 0.0431131
3: -0.0067881, 0.0667817, -0.0084666, 0.0703007, -0.0685304, 0.0670493
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0477590, 0.0012626, 0.0496467, -0.0467639, 0.0464964
6: -0.0369665, 0.0490958, -0.0386754, 0.0543882, -0.0913547, 0.0877712
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7034178, 0.9489263, 0.6882600, 0.9518959, -0.1995325, 0.2102108
9: 0.0471417, 0.0920409, 0.0455112, 0.0938498, -0.0467081, 0.0465297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0436411, 0.0592813, -0.1056784, 0.1020424
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050299, 0.0367439, -0.0031229, 0.0370446, -0.0420746, 0.0398668
3: -0.0067881, 0.0667817, -0.0043976, 0.0673565, -0.0663638, 0.0638044
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0477590, 0.0051905, 0.0480673, -0.0440732, 0.0425686
6: -0.0369665, 0.0490958, -0.0345330, 0.0499602, -0.0869267, 0.0836288
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301798, 0.0296620
8: 0.7034178, 0.9489263, 0.7009428, 0.9446972, -0.1971045, 0.2020483
9: 0.0471417, 0.0920409, 0.0494637, 0.0923364, -0.0451947, 0.0425772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0463971, 0.0584013, -0.0432869, 0.0601637, -0.1065608, 0.1016882
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308805
2: -0.0050299, 0.0367439, -0.0028783, 0.0373461, -0.0423760, 0.0396223
3: -0.0067881, 0.0667817, -0.0040907, 0.0679330, -0.0669383, 0.0637697
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0477590, 0.0054865, 0.0483766, -0.0437535, 0.0422725
6: -0.0369665, 0.0490958, -0.0342208, 0.0508273, -0.0877938, 0.0833166
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7034178, 0.9489263, 0.6984591, 0.9441545, -0.1983140, 0.2037709
9: 0.0471417, 0.0920409, 0.0497617, 0.0926327, -0.0454910, 0.0422793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0478678, 0.0645912, -0.1110106, 0.1033163
1: -0.0164326, 0.0162384, -0.0172537, 0.0171715, -0.0336040, 0.0334921
2: -0.0050456, 0.0357351, -0.0060478, 0.0388587, -0.0439044, 0.0417828
3: -0.0068073, 0.0648526, -0.0080637, 0.0708254, -0.0692218, 0.0650368
4: -0.0257235, -0.0011510, -0.0261833, -0.0005204, -0.0252031, 0.0250324
5: 0.0028639, 0.0467242, 0.0016516, 0.0499281, -0.0469098, 0.0450726
6: -0.0369866, 0.0461946, -0.0382654, 0.0551771, -0.0921637, 0.0844600
7: -0.0178716, 0.0135691, -0.0187933, 0.0142187, -0.0320902, 0.0323624
8: 0.7117278, 0.9489607, 0.6860011, 0.9511826, -0.1925888, 0.2129822
9: 0.0471227, 0.0910493, 0.0459027, 0.0941195, -0.0469968, 0.0451466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0478678, 0.0645912, -0.1109878, 0.1024290
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334776
2: -0.0050301, 0.0354300, -0.0060478, 0.0388587, -0.0438888, 0.0414778
3: -0.0067881, 0.0642716, -0.0080637, 0.0708254, -0.0690598, 0.0643206
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0464113, 0.0016516, 0.0499281, -0.0468019, 0.0447597
6: -0.0369667, 0.0453175, -0.0382654, 0.0551771, -0.0921438, 0.0835829
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320758, 0.0323523
8: 0.7142399, 0.9489262, 0.6860011, 0.9511826, -0.1888852, 0.2116635
9: 0.0471418, 0.0907734, 0.0459027, 0.0941195, -0.0469777, 0.0448707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0483328, 0.0637884, -0.1102077, 0.1037812
1: -0.0164326, 0.0162384, -0.0175171, 0.0174706, -0.0339032, 0.0337555
2: -0.0050456, 0.0357351, -0.0063692, 0.0385845, -0.0436301, 0.0421042
3: -0.0068073, 0.0648526, -0.0084666, 0.0703007, -0.0683667, 0.0651133
4: -0.0257235, -0.0011510, -0.0263309, -0.0003181, -0.0254053, 0.0251799
5: 0.0028639, 0.0467242, 0.0012626, 0.0496467, -0.0467829, 0.0454616
6: -0.0369866, 0.0461946, -0.0386754, 0.0543882, -0.0913748, 0.0848700
7: -0.0178716, 0.0135691, -0.0190889, 0.0144272, -0.0322988, 0.0326580
8: 0.7117278, 0.9489607, 0.6882600, 0.9518959, -0.1911783, 0.2084191
9: 0.0471227, 0.0910493, 0.0455112, 0.0938498, -0.0467272, 0.0455381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0436411, 0.0592813, -0.1057007, 0.0990895
1: -0.0164326, 0.0162384, -0.0148572, 0.0144485, -0.0308811, 0.0310956
2: -0.0050456, 0.0357351, -0.0031229, 0.0370446, -0.0420903, 0.0388579
3: -0.0068073, 0.0648526, -0.0043976, 0.0673565, -0.0662386, 0.0618742
4: -0.0257235, -0.0011510, -0.0248410, -0.0023608, -0.0233626, 0.0236900
5: 0.0028639, 0.0467242, 0.0051905, 0.0480673, -0.0439929, 0.0415337
6: -0.0369866, 0.0461946, -0.0345330, 0.0499602, -0.0869468, 0.0807276
7: -0.0178716, 0.0135691, -0.0161031, 0.0123225, -0.0301941, 0.0296722
8: 0.7117278, 0.9489607, 0.7009428, 0.9446972, -0.1887765, 0.2004530
9: 0.0471227, 0.0910493, 0.0494637, 0.0923364, -0.0452137, 0.0415856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0478678, 0.0645912, -0.1110106, 0.1033163
1: -0.0164326, 0.0162384, -0.0172537, 0.0171715, -0.0336040, 0.0334921
2: -0.0050456, 0.0357351, -0.0060478, 0.0388587, -0.0439044, 0.0417828
3: -0.0068073, 0.0648526, -0.0080637, 0.0708254, -0.0692218, 0.0650368
4: -0.0257235, -0.0011510, -0.0261833, -0.0005204, -0.0252031, 0.0250324
5: 0.0028639, 0.0467242, 0.0016516, 0.0499281, -0.0469098, 0.0450726
6: -0.0369866, 0.0461946, -0.0382654, 0.0551771, -0.0921637, 0.0844600
7: -0.0178716, 0.0135691, -0.0187933, 0.0142187, -0.0320902, 0.0323624
8: 0.7117278, 0.9489607, 0.6860011, 0.9511826, -0.1925888, 0.2129822
9: 0.0471227, 0.0910493, 0.0459027, 0.0941195, -0.0469968, 0.0451466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0483328, 0.0637884, -0.1102077, 0.1037812
1: -0.0164326, 0.0162384, -0.0175171, 0.0174706, -0.0339032, 0.0337555
2: -0.0050456, 0.0357351, -0.0063692, 0.0385845, -0.0436301, 0.0421042
3: -0.0068073, 0.0648526, -0.0084666, 0.0703007, -0.0683667, 0.0651133
4: -0.0257235, -0.0011510, -0.0263309, -0.0003181, -0.0254053, 0.0251799
5: 0.0028639, 0.0467242, 0.0012626, 0.0496467, -0.0467829, 0.0454616
6: -0.0369866, 0.0461946, -0.0386754, 0.0543882, -0.0913748, 0.0848700
7: -0.0178716, 0.0135691, -0.0190889, 0.0144272, -0.0322988, 0.0326580
8: 0.7117278, 0.9489607, 0.6882600, 0.9518959, -0.1911783, 0.2084191
9: 0.0471227, 0.0910493, 0.0455112, 0.0938498, -0.0467272, 0.0455381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0436411, 0.0592813, -0.1057007, 0.0990895
1: -0.0164326, 0.0162384, -0.0148572, 0.0144485, -0.0308811, 0.0310956
2: -0.0050456, 0.0357351, -0.0031229, 0.0370446, -0.0420903, 0.0388579
3: -0.0068073, 0.0648526, -0.0043976, 0.0673565, -0.0662386, 0.0618742
4: -0.0257235, -0.0011510, -0.0248410, -0.0023608, -0.0233626, 0.0236900
5: 0.0028639, 0.0467242, 0.0051905, 0.0480673, -0.0439929, 0.0415337
6: -0.0369866, 0.0461946, -0.0345330, 0.0499602, -0.0869468, 0.0807276
7: -0.0178716, 0.0135691, -0.0161031, 0.0123225, -0.0301941, 0.0296722
8: 0.7117278, 0.9489607, 0.7009428, 0.9446972, -0.1887765, 0.2004530
9: 0.0471227, 0.0910493, 0.0494637, 0.0923364, -0.0452137, 0.0415856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0478678, 0.0645912, -0.1110106, 0.1033163
1: -0.0164326, 0.0162384, -0.0172537, 0.0171715, -0.0336040, 0.0334921
2: -0.0050456, 0.0357351, -0.0060478, 0.0388587, -0.0439044, 0.0417828
3: -0.0068073, 0.0648526, -0.0080637, 0.0708254, -0.0692218, 0.0650368
4: -0.0257235, -0.0011510, -0.0261833, -0.0005204, -0.0252031, 0.0250324
5: 0.0028639, 0.0467242, 0.0016516, 0.0499281, -0.0469098, 0.0450726
6: -0.0369866, 0.0461946, -0.0382654, 0.0551771, -0.0921637, 0.0844600
7: -0.0178716, 0.0135691, -0.0187933, 0.0142187, -0.0320902, 0.0323624
8: 0.7117278, 0.9489607, 0.6860011, 0.9511826, -0.1925888, 0.2129822
9: 0.0471227, 0.0910493, 0.0459027, 0.0941195, -0.0469968, 0.0451466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0464194, 0.0554484, -0.0432869, 0.0601637, -0.1065831, 0.0987353
1: -0.0164326, 0.0162384, -0.0146567, 0.0142207, -0.0306533, 0.0308951
2: -0.0050456, 0.0357351, -0.0028783, 0.0373461, -0.0423917, 0.0386134
3: -0.0068073, 0.0648526, -0.0040907, 0.0679330, -0.0671164, 0.0618395
4: -0.0257235, -0.0011510, -0.0247287, -0.0025147, -0.0232087, 0.0235777
5: 0.0028639, 0.0467242, 0.0054865, 0.0483766, -0.0439335, 0.0412377
6: -0.0369866, 0.0461946, -0.0342208, 0.0508273, -0.0878139, 0.0804154
7: -0.0178716, 0.0135691, -0.0158781, 0.0121639, -0.0300355, 0.0294471
8: 0.7117278, 0.9489607, 0.6984591, 0.9441545, -0.1899860, 0.2052047
9: 0.0471227, 0.0910493, 0.0497617, 0.0926327, -0.0455101, 0.0412876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0483328, 0.0637884, -0.1101850, 0.1028940
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337410
2: -0.0050301, 0.0354300, -0.0063692, 0.0385845, -0.0436146, 0.0417992
3: -0.0067881, 0.0642716, -0.0084666, 0.0703007, -0.0685303, 0.0647877
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0464113, 0.0012626, 0.0496467, -0.0467640, 0.0451487
6: -0.0369667, 0.0453175, -0.0386754, 0.0543882, -0.0913549, 0.0839930
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7142399, 0.9489262, 0.6882600, 0.9518959, -0.1904554, 0.2102029
9: 0.0471418, 0.0907734, 0.0455112, 0.0938498, -0.0467080, 0.0452622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0436411, 0.0592813, -0.1056779, 0.0982022
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050301, 0.0354300, -0.0031229, 0.0370446, -0.0420747, 0.0385529
3: -0.0067881, 0.0642716, -0.0043976, 0.0673565, -0.0663624, 0.0615070
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0464113, 0.0051905, 0.0480673, -0.0440713, 0.0412208
6: -0.0369667, 0.0453175, -0.0345330, 0.0499602, -0.0869269, 0.0798505
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301797, 0.0296620
8: 0.7142399, 0.9489262, 0.7009428, 0.9446972, -0.1879029, 0.2020402
9: 0.0471418, 0.0907734, 0.0494637, 0.0923364, -0.0451946, 0.0413097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0478678, 0.0645912, -0.1109878, 0.1024290
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334776
2: -0.0050301, 0.0354300, -0.0060478, 0.0388587, -0.0438888, 0.0414778
3: -0.0067881, 0.0642716, -0.0080637, 0.0708254, -0.0690598, 0.0643206
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0464113, 0.0016516, 0.0499281, -0.0468019, 0.0447597
6: -0.0369667, 0.0453175, -0.0382654, 0.0551771, -0.0921438, 0.0835829
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320758, 0.0323523
8: 0.7142399, 0.9489262, 0.6860011, 0.9511826, -0.1888852, 0.2116635
9: 0.0471418, 0.0907734, 0.0459027, 0.0941195, -0.0469777, 0.0448707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0483328, 0.0637884, -0.1101850, 0.1028940
1: -0.0164198, 0.0162238, -0.0175171, 0.0174706, -0.0338904, 0.0337410
2: -0.0050301, 0.0354300, -0.0063692, 0.0385845, -0.0436146, 0.0417992
3: -0.0067881, 0.0642716, -0.0084666, 0.0703007, -0.0685303, 0.0647877
4: -0.0257162, -0.0011608, -0.0263309, -0.0003181, -0.0253981, 0.0251700
5: 0.0028828, 0.0464113, 0.0012626, 0.0496467, -0.0467640, 0.0451487
6: -0.0369667, 0.0453175, -0.0386754, 0.0543882, -0.0913549, 0.0839930
7: -0.0178572, 0.0135589, -0.0190889, 0.0144272, -0.0322844, 0.0326479
8: 0.7142399, 0.9489262, 0.6882600, 0.9518959, -0.1904554, 0.2102029
9: 0.0471418, 0.0907734, 0.0455112, 0.0938498, -0.0467080, 0.0452622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0436411, 0.0592813, -0.1056779, 0.0982022
1: -0.0164198, 0.0162238, -0.0148572, 0.0144485, -0.0308683, 0.0310810
2: -0.0050301, 0.0354300, -0.0031229, 0.0370446, -0.0420747, 0.0385529
3: -0.0067881, 0.0642716, -0.0043976, 0.0673565, -0.0663624, 0.0615070
4: -0.0257162, -0.0011608, -0.0248410, -0.0023608, -0.0233554, 0.0236802
5: 0.0028828, 0.0464113, 0.0051905, 0.0480673, -0.0440713, 0.0412208
6: -0.0369667, 0.0453175, -0.0345330, 0.0499602, -0.0869269, 0.0798505
7: -0.0178572, 0.0135589, -0.0161031, 0.0123225, -0.0301797, 0.0296620
8: 0.7142399, 0.9489262, 0.7009428, 0.9446972, -0.1879029, 0.2020402
9: 0.0471418, 0.0907734, 0.0494637, 0.0923364, -0.0451946, 0.0413097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0478678, 0.0645912, -0.1109878, 0.1024290
1: -0.0164198, 0.0162238, -0.0172537, 0.0171715, -0.0335913, 0.0334776
2: -0.0050301, 0.0354300, -0.0060478, 0.0388587, -0.0438888, 0.0414778
3: -0.0067881, 0.0642716, -0.0080637, 0.0708254, -0.0690598, 0.0643206
4: -0.0257162, -0.0011608, -0.0261833, -0.0005204, -0.0251958, 0.0250225
5: 0.0028828, 0.0464113, 0.0016516, 0.0499281, -0.0468019, 0.0447597
6: -0.0369667, 0.0453175, -0.0382654, 0.0551771, -0.0921438, 0.0835829
7: -0.0178572, 0.0135589, -0.0187933, 0.0142187, -0.0320758, 0.0323523
8: 0.7142399, 0.9489262, 0.6860011, 0.9511826, -0.1888852, 0.2116635
9: 0.0471418, 0.0907734, 0.0459027, 0.0941195, -0.0469777, 0.0448707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.53 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.53 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0463966, 0.0545612, -0.0432869, 0.0601637, -0.1065603, 0.0978480
1: -0.0164198, 0.0162238, -0.0146567, 0.0142207, -0.0306405, 0.0308806
2: -0.0050301, 0.0354300, -0.0028783, 0.0373461, -0.0423762, 0.0383083
3: -0.0067881, 0.0642716, -0.0040907, 0.0679330, -0.0669382, 0.0611126
4: -0.0257162, -0.0011608, -0.0247287, -0.0025147, -0.0232015, 0.0235678
5: 0.0028828, 0.0464113, 0.0054865, 0.0483766, -0.0437534, 0.0409248
6: -0.0369667, 0.0453175, -0.0342208, 0.0508273, -0.0877940, 0.0795383
7: -0.0178572, 0.0135589, -0.0158781, 0.0121639, -0.0300211, 0.0294370
8: 0.7142399, 0.9489262, 0.6984591, 0.9441545, -0.1864166, 0.2037711
9: 0.0471418, 0.0907734, 0.0497617, 0.0926327, -0.0454910, 0.0410118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483788, 0.0643713, -0.1076782, 0.1115581
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064011, 0.0387837, -0.0416758, 0.0447775
3: -0.0041086, 0.0699030, -0.0085063, 0.0706818, -0.0669177, 0.0701455
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054695, 0.0494333, 0.0012239, 0.0498511, -0.0443815, 0.0460978
6: -0.0342385, 0.0537900, -0.0387161, 0.0549611, -0.0891996, 0.0925061
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6899735, 0.9441853, 0.6866189, 0.9519659, -0.2126038, 0.2108355
9: 0.0497447, 0.0936454, 0.0454726, 0.0940456, -0.0443009, 0.0481727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0436862, 0.0600004, -0.1033072, 0.1068655
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0383764, -0.0031544, 0.0372902, -0.0401824, 0.0415308
3: -0.0041086, 0.0699030, -0.0044372, 0.0678261, -0.0648524, 0.0667820
4: -0.0247351, -0.0025060, -0.0248554, -0.0023410, -0.0223941, 0.0223494
5: 0.0054695, 0.0494333, 0.0051522, 0.0483193, -0.0414594, 0.0417284
6: -0.0342385, 0.0537900, -0.0345731, 0.0506667, -0.0849052, 0.0883631
7: -0.0158908, 0.0121729, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6899735, 0.9441853, 0.6989187, 0.9447667, -0.2095048, 0.2029891
9: 0.0497447, 0.0936454, 0.0494252, 0.0925778, -0.0428331, 0.0442201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0433297, 0.0612006, -0.1045078, 0.1071046
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289146
2: -0.0028921, 0.0385798, -0.0029075, 0.0377003, -0.0405924, 0.0414873
3: -0.0041084, 0.0702920, -0.0041280, 0.0686103, -0.0655686, 0.0671036
4: -0.0247350, -0.0024921, -0.0247422, -0.0024963, -0.0222387, 0.0222501
5: 0.0054697, 0.0496420, 0.0054508, 0.0487399, -0.0411208, 0.0416791
6: -0.0342384, 0.0543750, -0.0342584, 0.0518458, -0.0860841, 0.0886334
7: -0.0158908, 0.0121728, -0.0159051, 0.0121830, -0.0280738, 0.0280780
8: 0.6882984, 0.9441848, 0.6955412, 0.9442201, -0.2123580, 0.2057641
9: 0.0497448, 0.0938453, 0.0497257, 0.0929809, -0.0432361, 0.0441196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483788, 0.0643713, -0.1076782, 0.1115581
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064011, 0.0387837, -0.0416758, 0.0447775
3: -0.0041086, 0.0699030, -0.0085063, 0.0706818, -0.0669177, 0.0701455
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054695, 0.0494333, 0.0012239, 0.0498511, -0.0443815, 0.0460978
6: -0.0342385, 0.0537900, -0.0387161, 0.0549611, -0.0891996, 0.0925061
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6899735, 0.9441853, 0.6866189, 0.9519659, -0.2126038, 0.2108355
9: 0.0497447, 0.0936454, 0.0454726, 0.0940456, -0.0443009, 0.0481727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.55 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483788, 0.0643713, -0.1076782, 0.1115581
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064011, 0.0387837, -0.0416758, 0.0447775
3: -0.0041086, 0.0699030, -0.0085063, 0.0706818, -0.0669177, 0.0701455
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054695, 0.0494333, 0.0012239, 0.0498511, -0.0443815, 0.0460978
6: -0.0342385, 0.0537900, -0.0387161, 0.0549611, -0.0891996, 0.0925061
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6899735, 0.9441853, 0.6866189, 0.9519659, -0.2126038, 0.2108355
9: 0.0497447, 0.0936454, 0.0454726, 0.0940456, -0.0443009, 0.0481727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0436862, 0.0600004, -0.1033072, 0.1068655
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028921, 0.0383764, -0.0031544, 0.0372902, -0.0401824, 0.0415308
3: -0.0041086, 0.0699030, -0.0044372, 0.0678261, -0.0648524, 0.0667820
4: -0.0247351, -0.0025060, -0.0248554, -0.0023410, -0.0223941, 0.0223494
5: 0.0054695, 0.0494333, 0.0051522, 0.0483193, -0.0414594, 0.0417284
6: -0.0342385, 0.0537900, -0.0345731, 0.0506667, -0.0849052, 0.0883631
7: -0.0158908, 0.0121729, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6899735, 0.9441853, 0.6989187, 0.9447667, -0.2095048, 0.2029891
9: 0.0497447, 0.0936454, 0.0494252, 0.0925778, -0.0428331, 0.0442201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0433297, 0.0612006, -0.1045078, 0.1071046
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289146
2: -0.0028921, 0.0385798, -0.0029075, 0.0377003, -0.0405924, 0.0414873
3: -0.0041084, 0.0702920, -0.0041280, 0.0686103, -0.0655686, 0.0671036
4: -0.0247350, -0.0024921, -0.0247422, -0.0024963, -0.0222387, 0.0222501
5: 0.0054697, 0.0496420, 0.0054508, 0.0487399, -0.0411208, 0.0416791
6: -0.0342384, 0.0543750, -0.0342584, 0.0518458, -0.0860841, 0.0886334
7: -0.0158908, 0.0121728, -0.0159051, 0.0121830, -0.0280738, 0.0280780
8: 0.6882984, 0.9441848, 0.6955412, 0.9442201, -0.2123580, 0.2057641
9: 0.0497448, 0.0938453, 0.0497257, 0.0929809, -0.0432361, 0.0441196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0433069, 0.0631793, -0.0483788, 0.0643713, -0.1076782, 0.1115581
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321683, 0.0317768
2: -0.0028921, 0.0383764, -0.0064011, 0.0387837, -0.0416758, 0.0447775
3: -0.0041086, 0.0699030, -0.0085063, 0.0706818, -0.0669177, 0.0701455
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054695, 0.0494333, 0.0012239, 0.0498511, -0.0443815, 0.0460978
6: -0.0342385, 0.0537900, -0.0387161, 0.0549611, -0.0891996, 0.0925061
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6899735, 0.9441853, 0.6866189, 0.9519659, -0.2126038, 0.2108355
9: 0.0497447, 0.0936454, 0.0454726, 0.0940456, -0.0443009, 0.0481727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0433072, 0.0637749, -0.0479112, 0.0654786, -0.1087858, 0.1116861
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315117
2: -0.0028921, 0.0385798, -0.0060775, 0.0391620, -0.0420541, 0.0446573
3: -0.0041084, 0.0702920, -0.0081012, 0.0714051, -0.0675855, 0.0704261
4: -0.0247350, -0.0024921, -0.0261970, -0.0005017, -0.0242333, 0.0237049
5: 0.0054697, 0.0496420, 0.0016154, 0.0502390, -0.0441307, 0.0459906
6: -0.0342384, 0.0543750, -0.0383031, 0.0560488, -0.0902872, 0.0926781
7: -0.0158908, 0.0121728, -0.0188207, 0.0142381, -0.0301289, 0.0309935
8: 0.6882984, 0.9441848, 0.6835042, 0.9512478, -0.2156110, 0.2132404
9: 0.0497448, 0.0938453, 0.0458664, 0.0944174, -0.0446726, 0.0479789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0483788, 0.0643713, -0.1080337, 0.1079780
1: -0.0148696, 0.0144624, -0.0175432, 0.0175002, -0.0323698, 0.0320056
2: -0.0031379, 0.0371533, -0.0064011, 0.0387837, -0.0419216, 0.0435544
3: -0.0044166, 0.0675643, -0.0085063, 0.0706818, -0.0669786, 0.0677458
4: -0.0248479, -0.0023514, -0.0263454, -0.0002981, -0.0245498, 0.0239940
5: 0.0051722, 0.0481788, 0.0012239, 0.0498511, -0.0446789, 0.0458507
6: -0.0345522, 0.0502727, -0.0387161, 0.0549611, -0.0895134, 0.0889888
7: -0.0161169, 0.0123323, -0.0191182, 0.0144478, -0.0305647, 0.0314505
8: 0.7000468, 0.9447302, 0.6866189, 0.9519659, -0.2021253, 0.2096937
9: 0.0494456, 0.0924433, 0.0454726, 0.0940456, -0.0446001, 0.0469706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1212575
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0479112, 0.0654786, -0.1091410, 0.1075104
1: -0.0148696, 0.0144624, -0.0172781, 0.0171990, -0.0320686, 0.0317405
2: -0.0031379, 0.0371533, -0.0060775, 0.0391620, -0.0422999, 0.0432308
3: -0.0044166, 0.0675643, -0.0081012, 0.0714051, -0.0680443, 0.0676265
4: -0.0248479, -0.0023514, -0.0261970, -0.0005017, -0.0243462, 0.0238456
5: 0.0051722, 0.0481788, 0.0016154, 0.0502390, -0.0445277, 0.0456468
6: -0.0345522, 0.0502727, -0.0383031, 0.0560488, -0.0906010, 0.0885758
7: -0.0161169, 0.0123323, -0.0188207, 0.0142381, -0.0303550, 0.0311530
8: 0.7000468, 0.9447302, 0.6835042, 0.9512478, -0.2033772, 0.2150698
9: 0.0494456, 0.0924433, 0.0458664, 0.0944174, -0.0449719, 0.0465768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1212575
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0436862, 0.0600004, -0.1036627, 0.1032854
1: -0.0148696, 0.0144624, -0.0148830, 0.0144780, -0.0293476, 0.0293454
2: -0.0031379, 0.0371533, -0.0031544, 0.0372902, -0.0404281, 0.0403077
3: -0.0044166, 0.0675643, -0.0044372, 0.0678261, -0.0648879, 0.0643740
4: -0.0248479, -0.0023514, -0.0248554, -0.0023410, -0.0225069, 0.0225040
5: 0.0051722, 0.0481788, 0.0051522, 0.0483193, -0.0415780, 0.0415182
6: -0.0345522, 0.0502727, -0.0345731, 0.0506667, -0.0852190, 0.0848458
7: -0.0161169, 0.0123323, -0.0161321, 0.0123429, -0.0284598, 0.0284644
8: 0.7000468, 0.9447302, 0.6989187, 0.9447667, -0.1989958, 0.2017820
9: 0.0494456, 0.0924433, 0.0494252, 0.0925778, -0.0431323, 0.0430180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0433297, 0.0612006, -0.1048630, 0.1029289
1: -0.0148696, 0.0144624, -0.0146810, 0.0142479, -0.0291175, 0.0291434
2: -0.0031379, 0.0371533, -0.0029075, 0.0377003, -0.0408382, 0.0400608
3: -0.0044166, 0.0675643, -0.0041280, 0.0686103, -0.0659669, 0.0642941
4: -0.0248479, -0.0023514, -0.0247422, -0.0024963, -0.0223516, 0.0223908
5: 0.0051722, 0.0481788, 0.0054508, 0.0487399, -0.0416168, 0.0413705
6: -0.0345522, 0.0502727, -0.0342584, 0.0518458, -0.0863980, 0.0845311
7: -0.0161169, 0.0123323, -0.0159051, 0.0121830, -0.0282999, 0.0282374
8: 0.7000468, 0.9447302, 0.6955412, 0.9442201, -0.2000279, 0.2073538
9: 0.0494456, 0.0924433, 0.0497257, 0.0929809, -0.0435354, 0.0427175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0483788, 0.0643713, -0.1080337, 0.1079780
1: -0.0148696, 0.0144624, -0.0175432, 0.0175002, -0.0323698, 0.0320056
2: -0.0031379, 0.0371533, -0.0064011, 0.0387837, -0.0419216, 0.0435544
3: -0.0044166, 0.0675643, -0.0085063, 0.0706818, -0.0669786, 0.0677458
4: -0.0248479, -0.0023514, -0.0263454, -0.0002981, -0.0245498, 0.0239940
5: 0.0051722, 0.0481788, 0.0012239, 0.0498511, -0.0446789, 0.0458507
6: -0.0345522, 0.0502727, -0.0387161, 0.0549611, -0.0895134, 0.0889888
7: -0.0161169, 0.0123323, -0.0191182, 0.0144478, -0.0305647, 0.0314505
8: 0.7000468, 0.9447302, 0.6866189, 0.9519659, -0.2021253, 0.2096937
9: 0.0494456, 0.0924433, 0.0454726, 0.0940456, -0.0446001, 0.0469706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1212575
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0479112, 0.0654786, -0.1091410, 0.1075104
1: -0.0148696, 0.0144624, -0.0172781, 0.0171990, -0.0320686, 0.0317405
2: -0.0031379, 0.0371533, -0.0060775, 0.0391620, -0.0422999, 0.0432308
3: -0.0044166, 0.0675643, -0.0081012, 0.0714051, -0.0680443, 0.0676265
4: -0.0248479, -0.0023514, -0.0261970, -0.0005017, -0.0243462, 0.0238456
5: 0.0051722, 0.0481788, 0.0016154, 0.0502390, -0.0445277, 0.0456468
6: -0.0345522, 0.0502727, -0.0383031, 0.0560488, -0.0906010, 0.0885758
7: -0.0161169, 0.0123323, -0.0188207, 0.0142381, -0.0303550, 0.0311530
8: 0.7000468, 0.9447302, 0.6835042, 0.9512478, -0.2033772, 0.2150698
9: 0.0494456, 0.0924433, 0.0458664, 0.0944174, -0.0449719, 0.0465768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1212575
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0436862, 0.0600004, -0.1036627, 0.1032854
1: -0.0148696, 0.0144624, -0.0148830, 0.0144780, -0.0293476, 0.0293454
2: -0.0031379, 0.0371533, -0.0031544, 0.0372902, -0.0404281, 0.0403077
3: -0.0044166, 0.0675643, -0.0044372, 0.0678261, -0.0648879, 0.0643740
4: -0.0248479, -0.0023514, -0.0248554, -0.0023410, -0.0225069, 0.0225040
5: 0.0051722, 0.0481788, 0.0051522, 0.0483193, -0.0415780, 0.0415182
6: -0.0345522, 0.0502727, -0.0345731, 0.0506667, -0.0852190, 0.0848458
7: -0.0161169, 0.0123323, -0.0161321, 0.0123429, -0.0284598, 0.0284644
8: 0.7000468, 0.9447302, 0.6989187, 0.9447667, -0.1989958, 0.2017820
9: 0.0494456, 0.0924433, 0.0494252, 0.0925778, -0.0431323, 0.0430180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0436624, 0.0595992, -0.0433297, 0.0612006, -0.1048630, 0.1029289
1: -0.0148696, 0.0144624, -0.0146810, 0.0142479, -0.0291175, 0.0291434
2: -0.0031379, 0.0371533, -0.0029075, 0.0377003, -0.0408382, 0.0400608
3: -0.0044166, 0.0675643, -0.0041280, 0.0686103, -0.0659669, 0.0642941
4: -0.0248479, -0.0023514, -0.0247422, -0.0024963, -0.0223516, 0.0223908
5: 0.0051722, 0.0481788, 0.0054508, 0.0487399, -0.0416168, 0.0413705
6: -0.0345522, 0.0502727, -0.0342584, 0.0518458, -0.0863980, 0.0845311
7: -0.0161169, 0.0123323, -0.0159051, 0.0121830, -0.0282999, 0.0282374
8: 0.7000468, 0.9447302, 0.6955412, 0.9442201, -0.2000279, 0.2073538
9: 0.0494456, 0.0924433, 0.0497257, 0.0929809, -0.0435354, 0.0427175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0483788, 0.0643713, -0.1076781, 0.1090898
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321682, 0.0317768
2: -0.0028922, 0.0375331, -0.0064011, 0.0387837, -0.0416758, 0.0439342
3: -0.0041084, 0.0682905, -0.0085063, 0.0706818, -0.0669172, 0.0687420
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054697, 0.0485683, 0.0012239, 0.0498511, -0.0443814, 0.0458016
6: -0.0342384, 0.0513647, -0.0387161, 0.0549611, -0.0891995, 0.0900808
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6969191, 0.9441855, 0.6866189, 0.9519659, -0.2072566, 0.2108314
9: 0.0497447, 0.0928164, 0.0454726, 0.0940456, -0.0443009, 0.0473438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0479112, 0.0654786, -0.1087854, 0.1086222
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315116
2: -0.0028922, 0.0375331, -0.0060775, 0.0391620, -0.0420541, 0.0436105
3: -0.0041084, 0.0682905, -0.0081012, 0.0714051, -0.0675853, 0.0680267
4: -0.0247351, -0.0025060, -0.0261970, -0.0005017, -0.0242334, 0.0236909
5: 0.0054697, 0.0485683, 0.0016154, 0.0502390, -0.0441304, 0.0452747
6: -0.0342384, 0.0513647, -0.0383031, 0.0560488, -0.0902872, 0.0896678
7: -0.0158908, 0.0121729, -0.0188207, 0.0142381, -0.0301289, 0.0309936
8: 0.6969191, 0.9441855, 0.6835042, 0.9512478, -0.2042236, 0.2132380
9: 0.0497447, 0.0928164, 0.0458664, 0.0944174, -0.0446727, 0.0469500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0436862, 0.0600004, -0.1033072, 0.1043972
1: -0.0146681, 0.0142336, -0.0148830, 0.0144780, -0.0291461, 0.0291166
2: -0.0028922, 0.0375331, -0.0031544, 0.0372902, -0.0401824, 0.0406874
3: -0.0041084, 0.0682905, -0.0044372, 0.0678261, -0.0648521, 0.0653646
4: -0.0247351, -0.0025060, -0.0248554, -0.0023410, -0.0223941, 0.0223494
5: 0.0054697, 0.0485683, 0.0051522, 0.0483193, -0.0414588, 0.0414622
6: -0.0342384, 0.0513647, -0.0345731, 0.0506667, -0.0849051, 0.0859378
7: -0.0158908, 0.0121729, -0.0161321, 0.0123429, -0.0282337, 0.0283050
8: 0.6969191, 0.9441855, 0.6989187, 0.9447667, -0.2041392, 0.2029879
9: 0.0497447, 0.0928164, 0.0494252, 0.0925778, -0.0428331, 0.0433912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144016
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0433297, 0.0612006, -0.1045074, 0.1040407
1: -0.0146681, 0.0142336, -0.0146810, 0.0142479, -0.0289160, 0.0289145
2: -0.0028922, 0.0375331, -0.0029075, 0.0377003, -0.0405925, 0.0404406
3: -0.0041084, 0.0682905, -0.0041280, 0.0686103, -0.0655687, 0.0647458
4: -0.0247351, -0.0025060, -0.0247422, -0.0024963, -0.0222388, 0.0222361
5: 0.0054697, 0.0485683, 0.0054508, 0.0487399, -0.0411207, 0.0410181
6: -0.0342384, 0.0513647, -0.0342584, 0.0518458, -0.0860842, 0.0856232
7: -0.0158908, 0.0121729, -0.0159051, 0.0121830, -0.0280738, 0.0280781
8: 0.6969191, 0.9441855, 0.6955412, 0.9442201, -0.2012339, 0.2057638
9: 0.0497447, 0.0928164, 0.0497257, 0.0929809, -0.0432362, 0.0430907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144016
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0483788, 0.0643713, -0.1076781, 0.1090898
1: -0.0146681, 0.0142336, -0.0175432, 0.0175002, -0.0321682, 0.0317768
2: -0.0028922, 0.0375331, -0.0064011, 0.0387837, -0.0416758, 0.0439342
3: -0.0041084, 0.0682905, -0.0085063, 0.0706818, -0.0669172, 0.0687420
4: -0.0247351, -0.0025060, -0.0263454, -0.0002981, -0.0244370, 0.0238393
5: 0.0054697, 0.0485683, 0.0012239, 0.0498511, -0.0443814, 0.0458016
6: -0.0342384, 0.0513647, -0.0387161, 0.0549611, -0.0891995, 0.0900808
7: -0.0158908, 0.0121729, -0.0191182, 0.0144478, -0.0303386, 0.0312911
8: 0.6969191, 0.9441855, 0.6866189, 0.9519659, -0.2072566, 0.2108314
9: 0.0497447, 0.0928164, 0.0454726, 0.0940456, -0.0443009, 0.0473438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0433068, 0.0607110, -0.0479112, 0.0654786, -0.1087854, 0.1086222
1: -0.0146681, 0.0142336, -0.0172781, 0.0171990, -0.0318671, 0.0315116
2: -0.0028922, 0.0375331, -0.0060775, 0.0391620, -0.0420541, 0.0436105
3: -0.0041084, 0.0682905, -0.0081012, 0.0714051, -0.0675853, 0.0680267
4: -0.0247351, -0.0025060, -0.0261970, -0.0005017, -0.0242334, 0.0236909
5: 0.0054697, 0.0485683, 0.0016154, 0.0502390, -0.0441304, 0.0452747
6: -0.0342384, 0.0513647, -0.0383031, 0.0560488, -0.0902872, 0.0896678
7: -0.0158908, 0.0121729, -0.0188207, 0.0142381, -0.0301289, 0.0309936
8: 0.6969191, 0.9441855, 0.6835042, 0.9512478, -0.2042236, 0.2132380
9: 0.0497447, 0.0928164, 0.0458664, 0.0944174, -0.0446727, 0.0469500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
time: 0.55 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.64 seconds
IS_A1_B1_B1_B1_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1205018
IS_A1_B1_B1_B1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1203837
IS_A1_B1_B1_B1_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B1_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1026813
IS_A1_B1_B1_B1_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B1_B1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B1_B1_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B1_B1_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B1_B1_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B1_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B1_B2_A2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B1_B2_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1086967
IS_A1_B1_B2_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B1_B2_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1086967
IS_A1_B1_B2_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B1_B2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B1_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0919737, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B2_B1_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B2_B1_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B2_B1_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0902054, upper bound: 0.1083045
IS_A1_B2_B1_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B1_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A1_B2_B2_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1075840
IS_A1_B2_B2_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.0862903, upper bound: 0.1086641
IS_A2_B1_B1_B1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
IS_A2_B1_B1_B1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B1_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B1_B1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B1_B1_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B1_B1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1214615
IS_A2_B1_B1_B2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B1_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B1_B2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B1_B1_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B1_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B1_B1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1047672
IS_A2_B1_B1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1212575
IS_A2_B1_B2_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1212575
IS_A2_B1_B2_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1212575
IS_A2_B1_B2_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1212575
IS_A2_B1_B2_A1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B1_B2_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B2_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B2_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1212575
IS_A2_B1_B2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B1_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1218361, upper bound: 0.1057859
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1047672
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1149167, upper bound: 0.1057859
IS_A2_B2_A2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B2_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1144016, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144414
IS_A2_B2_A2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B2_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1144016
IS_A2_B2_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1047672, upper bound: 0.1149167
IS_A2_B2_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144026
IS_A2_B2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144016
IS_A2_B2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.64
Output dim: 8, lower bound: -0.1057859, upper bound: 0.1144026

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.62 + 598.27 = 600.90 seconds
